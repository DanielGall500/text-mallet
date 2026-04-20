from tmallet.obfuscators import (
    POSFilter,
    LemmaObfuscator,
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
    ShannonFilter,
)
from tmallet.utils import SpaCyInterface, LangConfig
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
    "noun-retain",  # part-of-speech filtering
    "noun-propn-retain",  # part-of-speech filtering
    "noun-remove",  # part-of-speech filtering
    "noun-propn-remove",  # part-of-speech filtering
    "scramble-hier-weak",  # dependency-parsing structural obfuscation
    "scramble-hier-strong",  # dependency-parsing structural obfuscation
    "scramble-BoW-sentence",  # randomly shuffle words at the sentence level
    "scramble-BoW-document",  # randomly shuffle words at the document level
    "shannon",  # filter based on an approximation of word importance
]


class TMallet:
    # apply_spacy_preprocessing: determines whether spacy is used or not
    # for the initial text processing
    # -> determined automatically based on the configuration selected
    apply_spacy_preprocessing: bool = False
    is_obfuscation_set_up: bool = False
    active_obfuscator = None
    active_config = None

    def __init__(self, lang: LangConfig = "en", prefer_gpu: bool = False):
        self.spacy_interface = SpaCyInterface(lang=lang, prefer_gpu=prefer_gpu)
        self.lang: LangConfig = lang
        self.device = "cuda" if prefer_gpu else "cpu"

    def obfuscate(self, text: Union[List[str], str]) -> Union[List[str], str]:
        if not self.active_obfuscator or not self.active_config:
            raise RuntimeError(
                "Please use `set_obfuscator` to setup the obfuscation details first."
            )

        if self.apply_spacy_preprocessing:
            text = self.spacy_interface.process(text)

        return self.active_obfuscator.obfuscate(text, config=self.active_config)

    def load_obfuscator(self, config: Dict):
        self.active_config = config
        algorithm = self.active_config["algorithm"]
        self.active_obfuscator = self._get_obfuscator(algorithm)
        return self

    def _get_obfuscator(
        self, algorithm: ObfuscationTechnique
    ) -> Union[Obfuscator, SpaCyObfuscator]:
        match algorithm:
            case "lemmatize":
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("lemma")
                return LemmaObfuscator()
            case (
                "noun-retain"
                | "noun-propn-retain"
                | "noun-remove"
                | "noun-propn-remove"
            ):
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("pos")
                return POSFilter()
            case "scramble-hier-weak" | "scramble-hier-strong":
                self.apply_spacy_preprocessing = True
                self.spacy_interface.set_pipeline("full")
                return HierarchicalScrambleObfuscator()
            case "scramble-BoW-sentence" | "scramble-BoW-document":
                self.apply_spacy_preprocessing = False
                return LinearScrambleObfuscator()
            case "shannon":
                self.apply_spacy_preprocessing = False
                self.spacy_interface.set_pipeline("pos")
                return ShannonFilter(
                    lang=self.lang,
                    spacy_interface=self.spacy_interface,
                    device=self.device,
                )
            case _:
                raise ValueError(
                    f"Input {algorithm} invalid. Please provide a valid obfuscation algorithm."
                )

    def _obfuscate_batch(
        self,
        batch,
        column: str,
        column_obfuscated: str,
        obfuscator: Union[Obfuscator | SpaCyObfuscator],
        config: Dict,
        multi: bool = True,
    ):
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
                # get the individual column suffix and obfuscated string
                for col, obfuscated_form in output.items():
                    key = f"{column_obfuscated}_{col}"

                    if key not in batch.keys():
                        batch[f"{column_obfuscated}_{col}"] = [obfuscated_form]
                    else:
                        batch[f"{column_obfuscated}_{col}"].append(obfuscated_form)
        return batch

    def obfuscate_dataset(
        self,
        dataset,
        column: str,
        column_obfuscated: str,
        config: Dict,
        batch_size: int = 10,
        num_proc: Optional[int] = None,
        device: str = "cuda",
    ):
        algorithm = config["algorithm"]
        obfuscator = self._get_obfuscator(algorithm, device)

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
        device: str = "cuda",
    ) -> None:
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
                    device=device,
                )

                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
        obfuscated_dataset = concatenate_datasets(processed_chunks)

        return obfuscated_dataset
