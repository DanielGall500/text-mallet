from tmallet.obfuscators import (
    ReplaceObfuscator,
    LemmaObfuscator,
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
    ShannonObfuscator,
)
from tmallet.utils import get_spacy_nlp
from datasets import load_from_disk, concatenate_datasets
from tmallet.obfuscators.base import Obfuscator, SpaCyObfuscator
from typing import Literal, Dict, Union, List, Optional
from functools import partial
from pathlib import Path
import torch
import os

torch.set_num_threads(1)

ObfuscationTechnique = Literal[
    "noun",
    "noun-pos",
    "no-noun",
    "no-noun-propn",
    "lemmatize",
    "scramble-BoW",
    "scramble-BoW-by-sentence",
    "scramble-shuffle-siblings",
    "scramble-reverse-head",
    "shannon",
]


class TMallet:
    def __init__(self):
        self.nlp = None

    def obfuscate(
            self, text: Union[List[str], str], config: Dict, device: str = "cpu"
    ) -> Union[List[str], str]:
        algorithm = config["algorithm"]
        obfuscator = self._get_obfuscator(algorithm, device)

        if self.nlp:
            text = self.nlp(text)

        return obfuscator.obfuscate(text, config=config)

    def _get_obfuscator(
        self, algorithm: ObfuscationTechnique, device
    ) -> Union[Obfuscator, SpaCyObfuscator]:
        prefer_gpu = (device == "cuda")

        match algorithm:
            case "noun" | "noun-propn" | "no-noun" | "no-noun-propn":
                self.nlp = get_spacy_nlp("ner", prefer_gpu=prefer_gpu)
                return ReplaceObfuscator(device=device)
            case "lemmatize":
                self.nlp = get_spacy_nlp("lemma", prefer_gpu=prefer_gpu)
                return LemmaObfuscator()
            case "scramble-BoW" | "scramble-BoW-by-sentence":
                self.nlp = None
                return LinearScrambleObfuscator(device=device)
            case "scramble-shuffle-siblings" | "scramble-reverse-head":
                self.nlp = get_spacy_nlp("full", prefer_gpu=prefer_gpu)
                return HierarchicalScrambleObfuscator(device=device)
            case "shannon":
                self.nlp = None
                return ShannonObfuscator(device=device)
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
        multi:bool=True
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
            obfuscation_output = [obfuscator.obfuscate(text,config=config) for text in texts]

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
        device: str = "cuda"
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
        device:str="cuda"
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
                    device=device
                )

                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
        obfuscated_dataset = concatenate_datasets(processed_chunks)

        return obfuscated_dataset
