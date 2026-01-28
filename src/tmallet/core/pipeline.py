from tmallet.obfuscators import (
    ReplaceObfuscator,
    LemmaObfuscator,
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
    SurprisalObfuscator,
    get_spacy_nlp,
)
from datasets import load_from_disk, concatenate_datasets
from tmallet.obfuscators.base import Obfuscator, SpaCyObfuscator
from typing import Literal,Dict,Union,List,Optional
from functools import partial
from pathlib import Path
import torch
import os

torch.set_num_threads(1)

ObfuscationTechnique = Literal[
    "noun",
    "noun-pos",
    "noun-propn",
    "noun-propn-pos",
    "lemmatization",
    "scramble-BoW",
    "scramble-BoW-by-sentence",
    "scramble-shuffle-siblings",
    "scramble-reverse-head",
    "mutual-information",
]


"""
Config Examples

config = {"algorithm": algorithm}
config = {"threshold": 4}
"""

class TMallet:
    def __init__(self):
        self.nlp = None

    def obfuscate(self, text: Union[List[str],str], config: Dict) -> Union[List[str],str]:
        algorithm = config["algorithm"]
        obfuscator = self._get_obfuscator(algorithm)
        return obfuscator.obfuscate(text, config)

    def _get_obfuscator(self, algorithm: ObfuscationTechnique) -> Union[Obfuscator,SpaCyObfuscator]:
        match algorithm:
            case "noun" | "noun-propn" | "noun-pos" | "noun-propn-pos":
                self.nlp = get_spacy_nlp("ner")
                return ReplaceObfuscator()
            case "lemmatization":
                self.nlp = get_spacy_nlp("lemma")
                return LemmaObfuscator()
            case "scramble-BoW" | "scramble-BoW-by-sentence":
                return LinearScrambleObfuscator()
            case "scramble-shuffle-siblings" | "scramble-reverse-head":
                self.nlp = get_spacy_nlp("full")
                return HierarchicalScrambleObfuscator()
            case "mutual-information":
                return SurprisalObfuscator()
            case _:
                raise ValueError("Please provide a valid obfuscation algorithm.")

    def _obfuscate_batch(
            self, batch, column: str, column_obfuscated: str, 
            obfuscator: Union[Obfuscator|SpaCyObfuscator], 
            config: Dict, 
    ):
        if column not in batch.columns:
            raise KeyError(f"Invalid column provided. Please choose one of {batch.columns}")
        texts = batch[column]

        is_using_spacy = issubclass(type(obfuscator), SpaCyObfuscator)
        if is_using_spacy:
            texts = self.nlp.pipe(texts)

        batch[column_obfuscated] = [
            obfuscator.obfuscate(text, config=config) for text in texts
        ]
        return batch

    def obfuscate_dataset(
        self,
        dataset,
        column: str,
        column_obfuscated: str,
        config: Dict,
        batch_size: int = 100,
        num_proc:Optional[int]=None
    ):
        obfuscated_dataset = dataset.map(
            partial(
                self._obfuscate_batch,
                column=column,
                column_obfuscated=column_obfuscated,
                config=config,
            ),
            batched=True,
            batch_size=batch_size,
            desc="Obfuscating...",
            num_proc=num_proc,
            cache_file_name=None,
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
        num_proc:Optional[int]=None
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
                    num_proc=num_proc
                )

                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
        obfuscated_dataset = concatenate_datasets(processed_chunks)

        return obfuscated_dataset
