from tmallet.obfuscators import (
    ReplaceObfuscator,
    LemmaObfuscator,
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
    SurprisalObfuscator,
    get_spacy_nlp,
)
from tmallet.obfuscators.base import SpaCyObfuscator
from datasets import load_from_disk, concatenate_datasets
from functools import partial
from pathlib import Path
from typing import Literal
import os
import torch

torch.set_num_threads(1)

nlp = get_spacy_nlp("full")
replace_obfus = ReplaceObfuscator()
lemma_obfus = LemmaObfuscator()

hierarchical_scramble_obfus = HierarchicalScrambleObfuscator()
linear_scramble_obfus = LinearScrambleObfuscator()

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
    "surprisal",
]


def obfuscate_batch(
    batch, algorithm: ObfuscationTechnique, obfuscated_column_name: str
):
    texts = batch["text"]

    match algorithm:
        case "noun" | "noun-propn" | "noun-pos" | "noun-propn-pos":
            obfuscator = ReplaceObfuscator()
        case "lemmatization":
            obfuscator = LemmaObfuscator()
        case "scramble-BoW" | "scramble-BoW-by-sentence":
            obfuscator = LinearScrambleObfuscator()
            config = {"algorithm": algorithm}
        case "scramble-shuffle-siblings" | "scramble-reverse-head":
            obfuscator = HierarchicalScrambleObfuscator()
            config = {"algorithm": algorithm}
        case "surprisal":
            obfuscator = SurprisalObfuscator()
            config = {"threshold": 4}

    is_using_spacy = issubclass(type(obfuscator), SpaCyObfuscator)
    if is_using_spacy:
        texts = nlp.pipe(texts)

    batch[obfuscated_column_name] = [
        obfuscator.obfuscate(text, config=config) for text in texts
    ]

    return batch


class TextMallet:
    def run(
        self,
        dataset,
        algorithm: ObfuscationTechnique,
        obfuscated_column_name: str,
        batch_size: int = 100,
    ):
        obfuscated_dataset = dataset.map(
            partial(
                obfuscate_batch,
                algorithm=algorithm,
                obfuscated_column_name=obfuscated_column_name,
            ),
            batched=True,
            batch_size=batch_size,
            desc="Obfuscating...",
            num_proc=None,
            cache_file_name=None,
            load_from_cache_file=False,
        )
        return obfuscated_dataset

    def run_in_chunks(
        self,
        dataset,
        algorithm: ObfuscationTechnique,
        obfuscated_column_name: str,
        save_chunks_to_folder: Path,
        batch_size: int = 100,
        chunk_size: int = 5_000,
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

                chunk = chunk.map(
                    partial(
                        obfuscate_batch,
                        algorithm=algorithm,
                        obfuscated_column_name=obfuscated_column_name,
                    ),
                    batched=True,
                    batch_size=batch_size,
                    desc=f"Obfuscating [{start}:{end}]...",
                    num_proc=None,
                    cache_file_name=None,
                    load_from_cache_file=False,
                )

                chunk.save_to_disk(ckpt_path)

            processed_chunks.append(chunk)
        obfuscated_dataset = concatenate_datasets(processed_chunks)

        return obfuscated_dataset
