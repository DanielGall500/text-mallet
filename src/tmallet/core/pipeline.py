from tmallet.obfuscators import (
    POSFilter,
    POSFilterConfig,
    LemmaObfuscator,
    LinearScrambleObfuscator,
    LinearScrambleConfig,
    HierarchicalScrambleObfuscator,
    HierarchicalScrambleConfig,
    ShannonFilter,
    ShannonFilterConfig,
)
from tmallet.utils import SpaCyInterface, LangConfig, flatten_dict
from datasets import load_from_disk, concatenate_datasets
from tmallet.obfuscators.base import Obfuscator, SpaCyObfuscator
from typing import Literal, Dict, Union, List, Optional
from functools import partial
from pathlib import Path
import torch
import os

torch.set_num_threads(1)

ObfuscationTechnique = Literal[
    "lemmatize",  # convert words to their roots
    "pos-filter" "scramble-hier",  # dependency-parsing structural obfuscation
    "scramble-BoW",  # randomly shuffle words at the sentence or document level
    "shannon",  # filter based on an approximation of word importance
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
    active_config = None

    def __init__(self, lang: LangConfig = "en", prefer_gpu: bool = False):
        """Initialises the TMallet pipeline.

        Args:
            lang (LangConfig, optional): The target language configuration - either "en" (English) or "de" (German). Defaults to "en". Let us know if you'd be interested in support for further languages.
            prefer_gpu (bool, optional): If True, attempts to allocate spaCy operations 
                on the GPU. Defaults to False.
        """
        self.spacy_interface = SpaCyInterface(lang=lang, prefer_gpu=prefer_gpu)
        self.lang: LangConfig = lang
        self.prefer_gpu = prefer_gpu

    def load_obfuscator(self, algorithm: str, config: Dict):
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
        return self

    def obfuscate(self, text: Union[List[str], str]) -> Union[List[str], str]:
        """Obfuscates standalone text strings or lists of strings.

        Requires an obfuscator to be loaded via `load_obfuscator` prior to invocation.

        Args:
            text (Union[List[str], str]): Single text payload or collection of texts to process.

        Raises:
            RuntimeError: If an obfuscator and configuration have not been loaded yet.

        Returns:
            Union[List[str], str]: The modified, obfuscated text or collection of texts.
        """
        if not self.active_obfuscator or not self.active_config:
            raise RuntimeError(
                "Please use `set_obfuscator` to setup the obfuscation details first."
            )

        if self.apply_spacy_preprocessing:
            text = self.spacy_interface.process(text)

        return self.active_obfuscator.obfuscate(text)

    def _obfuscate_batch(
        self,
        batch,
        column: str,
        column_obfuscated: str,
        obfuscator: Union[Obfuscator | SpaCyObfuscator],
        config: Dict,
        multi: bool = True,
    ):
        """Processes a single dictionary batch extracted from a Dataset pipeline wrapper.

        Args:
            batch (Dict[str, Any]): A batch slice containing lists mapped to column keys.
            column (str): The column key containing the raw text strings.
            column_obfuscated (str): Target base column key for saving the output.
            obfuscator (Union[Obfuscator, SpaCyObfuscator]): Instantiated algorithm engine.
            config (Dict): Parameters required for text conversion.
            multi (bool, optional): If True, flattens a complex nested dictionary output 
                directly into the batch root elements. Defaults to True.

        Raises:
            KeyError: If the specified target data column does not exist inside the batch.

        Returns:
            Dict[str, Any]: The batch containing the obfuscated results.
        """
        if column not in batch.keys():
            raise KeyError(
                f"Invalid column provided. Please choose one of {batch.columns}"
            )
        texts = batch[column]

        is_using_spacy = issubclass(type(obfuscator), SpaCyObfuscator)
        if is_using_spacy:
            texts = self.nlp.pipe(texts)

        if not multi:
            batch[column_obfuscated] = [
                obfuscator.obfuscate(text, config=config) for text in texts
            ]
        else:
            # a list of dictionaries, each containing the obfuscated formats under a key
            obfuscation_output = [
                obfuscator.obfuscate(text, config=config) for text in texts
            ]

            # for each dictionary (one per sample)
            for output in obfuscation_output:
                flatten = flatten_dict(output)
                batch.update(flatten)

        return batch

    def obfuscate_dataset(
        self,
        dataset,
        column: str,
        column_obfuscated: str,
        config: Dict,
        batch_size: int = 10,
        num_proc: Optional[int] = None,
    ):
        """Maps obfuscation across an entire HuggingFace/compatible dataset object sequentially.

        Args:
            dataset (Dataset): The underlying dataset collection containing columns of data.
            column (str): Key of the column containing raw target text.
            column_obfuscated (str): Target base column key for saving the output.
            config (Dict): Configuration properties outlining parameters and algorithm name.
            batch_size (int, optional): Size of chunk arrays processed together. Defaults to 10.
            num_proc (Optional[int], optional): CPU core count split handling parallel tasks. 
                Defaults to None.

        Returns:
            Dataset: A newly updated copy of the dataset containing obfuscation columns.
        """
        algorithm = config["algorithm"]
        obfuscator = self._get_obfuscator(algorithm)

        obfuscated_dataset = dataset.map(
            partial(
                self._obfuscate_batch,
                column=column,
                column_obfuscated=column_obfuscated,
                obfuscator=obfuscator,
                config=config,
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
        dataset,
        column: str,
        column_obfuscated: str,
        config: Dict,
        save_chunks_to_folder: Path,
        chunk_size: int = 5_000,
        batch_size: int = 100,
        num_proc: Optional[int] = None,
    ) -> None:
        """Processes a large dataset by slicing it into manageable chunks saved to disk.

        Ensures fault tolerance by loading existing saved cache checkpoints 
        from a folder structure if a massive operation was interrupted midway.

        Args:
            dataset (Dataset): The large input dataset collection.
            column (str): Key of the column containing raw target text.
            column_obfuscated (str): Target base column key for saving the output.
            config (Dict): Configuration properties outlining parameters and algorithm name.
            save_chunks_to_folder (Path): System path directory to log or load disk checkpoints.
            chunk_size (int, optional): Slices of dataset mapped out during step intervals. 
                Defaults to 5_000.
            batch_size (int, optional): Inner array size configuration mapped to `.map`. 
                Defaults to 100.
            num_proc (Optional[int], optional): CPU processing core parallelism configuration limit. 
                Defaults to None.

        Returns:
            Dataset: A unified concatenated dataset collection composed of processed fragments.
        """
        processed_chunks = []
        num_samples = len(dataset)

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            ckpt_path = Path(save_chunks_to_folder) / f"obfuscated_ckpt_{start}_{end}"

            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint {ckpt_path}")
                chunk = load_from_disk(ckpt_path)
            else:
                print(f"Processing examples {start}:{end}")
                chunk = dataset.select(range(start, end))

                chunk = self.obfuscate_dataset(
                    chunk,
                    column=column,
                    column_obfuscated=column_obfuscated,
                    config=config,
                    batch_size=batch_size,
                    num_proc=num_proc,
                )

                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
        obfuscated_dataset = concatenate_datasets(processed_chunks)

        return obfuscated_dataset

    def get_active_obfuscator(self):
        return self.active_obfuscator

    def _validate_config(self, algorithm: str, config: Dict):
        match algorithm:
            case "pos-filter":
                return POSFilterConfig(**config)
            case "scramble-BoW":
                return LinearScrambleConfig(**config)
            case "scramble-hier":
                return HierarchicalScrambleConfig(**config)
            case "shannon":
                return ShannonFilterConfig(**config)

    def _get_obfuscator(
        self, algorithm: ObfuscationTechnique
    ) -> Union[Obfuscator, SpaCyObfuscator]:
        match algorithm:
            case "lemmatize":
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("lemma")
                return LemmaObfuscator()
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
            case _:
                raise ValueError(
                    f"Input {algorithm} invalid. Please provide a valid obfuscation algorithm."
                )

