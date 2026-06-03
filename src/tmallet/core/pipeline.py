import os
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Literal

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from tmallet.obfuscators import (
    HierarchicalScrambleConfig,
    HierarchicalScrambleObfuscator,
    LemmaObfuscator,
    LinearScrambleConfig,
    LinearScrambleObfuscator,
    POSFilter,
    POSFilterConfig,
    ShannonFilter,
    ShannonFilterConfig,
)
from tmallet.obfuscators.base import Obfuscator, SpaCyObfuscator
from tmallet.utils import LangConfig, SpaCyInterface, flatten_dict

ObfuscationTechnique = Literal[
    "pos-filter",  # retain or remove specific POS tags
    "scramble-hier",  # dependency-parsing structural obfuscation
    "scramble-BoW",  # randomly shuffle words at the sentence or document level
    "shannon",  # filter based on an approximation of word importance
    "lemmatize",  # word-level lemmatisation - still available, but no longer supported
]


class TMallet:
    """A text obfuscation manager that applies transformations to text.

    This class applies selected algorithmic obfuscation techniques (such as POS filtering,
    bag-of-words scrambling, or information-theoretic filtering) to strings, lists of text,
    or entire datasets.

    Arguments:
        lang (LangConfig): The language configuration code (e.g., "en").
        prefer_gpu (bool): Whether spaCy is configured to leverage GPU acceleration.
    """

    # apply_spacy_preprocessing: determines whether spacy is used or not
    # for the initial text processing
    # -> determined automatically based on the configuration selected
    apply_spacy_preprocessing: bool = False
    is_obfuscation_set_up: bool = False
    active_obfuscator = None
    active_config: dict | None = None
    active_algorithm: str | None = None

    def __init__(self, lang: LangConfig = "en", prefer_gpu: bool = False):
        """Initialises the obfuscation pipeline.

        Args:
            lang (LangConfig, optional): The target language configuration - either "en" (English) or "de" (German). Defaults to "en". Let us know if you'd be interested in support for further languages.
            prefer_gpu (bool, optional): If True, attempts to allocate spaCy operations
                on the GPU. Defaults to False.
        """
        self.spacy_interface: SpaCyInterface = SpaCyInterface(
            lang=lang, prefer_gpu=prefer_gpu
        )
        self.lang: LangConfig = lang
        self.prefer_gpu = prefer_gpu

    def load_obfuscator(
        self,
        algorithm: ObfuscationTechnique,
        config: dict[str, str],
    ):
        """Validates configuration and dynamically instantiates an obfuscation algorithm.

        Args:
            algorithm (str): The identifier of the obfuscation technique (e.g., 'pos-filter').
            config (Dict): Key-value pairings containing parameters for the specific algorithm.

        Returns:
            TMallet: The current class instance to allow for method chaining.
        """
        self.active_config = self._validate_config(algorithm, config)
        self.active_obfuscator = self._get_obfuscator(algorithm)
        self.active_obfuscator.set_config(self.active_config)
        self.active_algorithm = algorithm
        return self

    def obfuscate(self, text: str) -> dict | str:
        """Obfuscates standalone text strings or lists of strings.

        Requires an obfuscator to be loaded via `load_obfuscator` prior to invocation.

        Args:
            text (Union[List[str], str]): Single text payload or collection of texts to process.

        Raises:
            RuntimeError: If an obfuscator and configuration have not been loaded yet.

        Returns:
            Dict: The modified, obfuscated text or collection of texts in the form of a dictionary.
        """
        if (
            self.active_obfuscator is None
            and self.active_config is None
            and not self.active_algorithm == "lemmatize"
        ):
            raise RuntimeError(
                "Please use `set_obfuscator` to setup the obfuscation details first."
            )

        if self.apply_spacy_preprocessing:
            text = self.spacy_interface.process(text)

        return self.active_obfuscator.obfuscate(text)

    def _obfuscate_batch(
        self,
        batch: dict,
        column: str,
        column_obfuscated: str,
        multi: bool = True,
    ) -> dict:
        """Processes a single dictionary batch extracted from a Dataset pipeline wrapper.

        Args:
            batch (Dict[str, Any]): A batch slice containing lists mapped to column keys.
            column (str): The column key containing the raw text strings.
            column_obfuscated (str): Target base column key for saving the output.
            multi (bool, optional): If True, flattens a complex nested dictionary output
                directly into the batch root elements. Defaults to True.

        Raises:
            KeyError: If the specified target data column does not exist inside the batch.

        Returns:
            Dict[str, Any]: The batch containing the obfuscated results.
        """
        if column not in batch.keys():
            raise KeyError(
                f"Invalid column provided. Please choose one of {list(batch.keys())}"
            )
        texts = batch[column]

        if not multi:
            batch[column_obfuscated] = [self.obfuscate(text) for text in texts]
        else:
            obfuscation_output = [flatten_dict(self.obfuscate(text)) for text in texts]

            all_keys = obfuscation_output[0].keys()
            batch.update(
                {
                    key: [sample[key] for sample in obfuscation_output]
                    for key in all_keys
                }
            )

        return batch

    def obfuscate_dataset(
        self,
        dataset: Dataset,
        column: str,
        column_obfuscated: str,
        batch_size: int = 10,
        num_proc: int | None = None,
    ):
        """Maps obfuscation across an entire HuggingFace/compatible dataset object sequentially.

        Args:
            dataset (Dataset): The underlying dataset collection containing columns of data.
            column (str): Key of the column containing raw target text.
            column_obfuscated (str): Target base column key for saving the output.
            batch_size (int, optional): Size of chunk arrays processed together. Defaults to 10.
            num_proc (Optional[int], optional): CPU core count split handling parallel tasks.
                Defaults to None.

        Returns:
            Dataset: A newly updated copy of the dataset containing obfuscation columns.
        """
        obfuscated_dataset = dataset.map(
            partial(
                self._obfuscate_batch,
                column=column,
                column_obfuscated=column_obfuscated,
            ),
            batched=True,
            batch_size=batch_size,
            desc="Obfuscating...",
            num_proc=num_proc,
            # cache_file_name=None,
            load_from_cache_file=False,
        )
        return obfuscated_dataset

    def obfuscate_dataset_by_chunk(
        self,
        dataset_repo: str,
        column: str,
        column_obfuscated: str,
        save_chunks_to_folder: Path,
        dataset_config: str | None = None,
        dataset_split: str = "train",
        chunk_size: int = 5_000,
        batch_size: int = 100,
        num_proc: int | None = None,
        num_samples: int | None = None,
    ) -> Dataset:
        """Streams a dataset from the Hub in chunks, obfuscates each chunk,
        and saves checkpoints to disk for fault tolerance.

        Args:
            dataset_repo (str): HuggingFace Hub repo ID or local path for load_dataset.
            column (str): Key of the column containing raw target text.
            column_obfuscated (str): Target base column key for saving the output.
            save_chunks_to_folder (Path): Directory to save/load disk checkpoints.
            dataset_config (Optional[str]): Dataset config/subset name passed to load_dataset.
            dataset_split (str): Split to stream (e.g. "train", "validation"). Defaults to "train".
            chunk_size (int): Number of examples per chunk. Defaults to 5_000.
            batch_size (int): Inner batch size passed to .map. Defaults to 100.
            num_proc (Optional[int]): CPU parallelism for .map. Defaults to None.
            num_samples (Optional[int]): Optional cap on total examples to process.

        Returns:
            Dataset: Concatenated dataset of all processed chunks.
        """
        stream = load_dataset(
            dataset_repo,
            dataset_config,
            split=dataset_split,
            streaming=True,
        )
        if num_samples:
            stream = stream.take(num_samples)

        iterator = iter(stream)
        processed_chunks = []
        chunk_index = 0

        while True:
            start = chunk_index * chunk_size
            end = start + chunk_size
            ckpt_path = Path(save_chunks_to_folder) / f"obfuscated_ckpt_{start}_{end}"

            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint {ckpt_path}")
                chunk = load_from_disk(ckpt_path)
                # Advance stream past already-processed examples
                list(islice(iterator, chunk_size))
            else:
                rows = list(islice(iterator, chunk_size))
                if not rows:
                    break  # Stream exhausted
                print(f"Processing examples {start}:{end}")
                chunk = Dataset.from_list(rows)
                chunk = self.obfuscate_dataset(
                    chunk,
                    column=column,
                    column_obfuscated=column_obfuscated,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )
                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
            chunk_index += 1

        obfuscated_dataset = concatenate_datasets(processed_chunks)
        return obfuscated_dataset

    def get_active_obfuscator(self):
        return self.active_obfuscator

    def _validate_config(self, algorithm: str, config: dict):
        match algorithm:
            case "pos-filter":
                return POSFilterConfig(**config)
            case "scramble-BoW":
                return LinearScrambleConfig(**config)
            case "scramble-hier":
                return HierarchicalScrambleConfig(**config)
            case "shannon":
                return ShannonFilterConfig(**config)
            case "lemmatize":
                return None

    def _get_obfuscator(
        self, algorithm: ObfuscationTechnique
    ) -> Obfuscator | SpaCyObfuscator:
        match algorithm:
            case "pos-filter":
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("pos")
                return POSFilter()
            case "scramble-hier":
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("full")
                return HierarchicalScrambleObfuscator()
            case "scramble-BoW":
                self.apply_spacy_preprocessing = False
                return LinearScrambleObfuscator()
            case "shannon":
                self.apply_spacy_preprocessing = False
                self.spacy_interface.set_pipeline("pos")
                return ShannonFilter(
                    lang=self.lang,
                    spacy_interface=self.spacy_interface,
                    prefer_gpu=self.prefer_gpu,
                )
            case "lemmatize":
                self.apply_spacy_preprocessing = True
                return LemmaObfuscator()
            case _:
                raise ValueError(
                    f"Input {algorithm} invalid. Please provide a valid obfuscation algorithm."
                )
