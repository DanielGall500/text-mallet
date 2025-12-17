from pandas.core.series import algorithms
from reef.dataloaders.txt_loader import TxtLoader
from reef.obfuscators.replace import ReplaceObfuscator
from reef.obfuscators.lemmatise import LemmaObfuscator
from reef.obfuscators.scramble import (
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
)
from datasets import load_from_disk, load_dataset, DatasetDict, concatenate_datasets
from reef.obfuscators.spacy_registry import get_spacy_nlp
import os
import torch

torch.set_num_threads(1)

nlp = get_spacy_nlp("ner")
replace_obfus = ReplaceObfuscator()
MAP_BATCH_SIZE = 10     # spaCy batch size
def process_batch(batch):
    texts = batch["text"]

    docs = nlp.pipe(
        texts,
        batch_size=MAP_BATCH_SIZE,
        n_process=2
    )

    batch["text_lemmas"] = [
        replace_obfus.obfuscate(doc, algorithm="nouns-only", replace_with_pos=True) for doc in docs
    ]
    return batch

class ReefPipeline:
    def run(self, repo_name: str) -> None:
        dataset = load_dataset(repo_name)


        CHECKPOINT_DIR = "new_checkpoints/replace_nouns_with_pos"
        FINAL_DIR = "new_checkpoints/transformed_dataset_replace/"
        CHUNK_SIZE = 5_000       # examples per checkpoint

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        final_splits = {}

        for split_name, split_ds in dataset.items():
            print(f"\nProcessing split: {split_name}")
            processed_chunks = []

            for start in range(0, len(split_ds), CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, len(split_ds))
                ckpt_path = f"{CHECKPOINT_DIR}/{split_name}_{start}"

                if os.path.exists(ckpt_path):
                    print(f"  ↳ Loading checkpoint {ckpt_path}")
                    chunk = load_from_disk(ckpt_path)
                else:
                    print(f"  ↳ Processing examples {start}:{end}")
                    chunk = split_ds.select(range(start, end))

                    chunk = chunk.map(
                        process_batch,
                        batched=True,
                        batch_size=MAP_BATCH_SIZE,
                        desc=f"Lemmatising {split_name} [{start}:{end}]",
                        num_proc=8,
                        cache_file_name=None
                    )

                    chunk.save_to_disk(ckpt_path)

                processed_chunks.append(chunk)

            final_splits[split_name] = concatenate_datasets(processed_chunks)

        # Reassemble DatasetDict
        from datasets import DatasetDict
        final_dataset = DatasetDict(final_splits)

        final_dataset.save_to_disk(FINAL_DIR)


    def run_test(self) -> bool:
        path_to_dataset = "./datasets/leipzig.txt"

        txt_loader = TxtLoader()
        replace_obfuscator = ReplaceObfuscator()

        text = txt_loader.load(path_to_dataset)

        lemma_obf = LemmaObfuscator()
        output_lemma = lemma_obf.obfuscate(text)

        linear_obfus = LinearScrambleObfuscator()
        output_linear = linear_obfus.obfuscate(text)

        hierarchical_obfus = HierarchicalScrambleObfuscator()
        output_shuffle_siblings = hierarchical_obfus.obfuscate(text, "shuffle-siblings")
        output_rev_head_direction = hierarchical_obfus.obfuscate(
            text, "reverse-head-direction"
        )

        replace_obfus = ReplaceObfuscator()
        output_nouns_and_propn_only = replace_obfus.obfuscate(
            text, algorithm="nouns-and-prop-only"
        )
        output_nouns_only = replace_obfus.obfuscate(
                text, algorithm="nouns-only"
        )
        output_no_nouns_or_propn = replace_obfus.obfuscate(
            text, algorithm="no-nouns-or-prop"
        )
        output_no_nouns = replace_obfus.obfuscate(
                text, 
                algorithm="no-nouns"
        )

        output_nouns_and_propn_only_replace = replace_obfus.obfuscate(
            text, 
            algorithm="nouns-and-prop-only",
            replace_with_pos=True
        )
        output_nouns_only_replace = replace_obfus.obfuscate(
                text,
                algorithm="nouns-only",
                replace_with_pos=True
        )
        output_no_nouns_or_propn_replace = replace_obfus.obfuscate(
                text, 
                algorithm="no-nouns-or-prop",
                replace_with_pos=True
        )
        output_no_nouns_replace = replace_obfus.obfuscate(
                text, 
                algorithm="no-nouns",
                replace_with_pos=True
        )

        obfuscations = {
            "Lemmas Only": output_lemma,
            "Linear": output_linear,
            "Hierarchical (Shuffle Siblings Randomly)": output_shuffle_siblings,
            "Hierarchical (Reverse Head Direction Randomly)": output_rev_head_direction,
            "Nouns & Proper Nouns Only": output_nouns_and_propn_only,
            "Nouns Only": output_nouns_only,
            "No Nouns": output_no_nouns,
            "No Nouns Nor Proper Nouns": output_no_nouns_or_propn,
            "Nouns & Proper Nouns Only (incl. POS)": output_nouns_and_propn_only_replace,
            "Nouns Only (incl. POS)": output_nouns_only_replace,
            "No Nouns (incl. POS)": output_no_nouns_replace,
            "No Nouns Nor Proper Nouns (incl. POS)": output_no_nouns_or_propn_replace
        }

        # Display results
        print("=" * 80)
        print("ORIGINAL TEXT")
        print("=" * 80)
        print(text)
        print("\n")

        for method_name, obfuscated_text in obfuscations.items():
            print("=" * 80)
            print(f"OBFUSCATION METHOD: {method_name}")
            print("=" * 80)
            print(obfuscated_text)
            print("\n")

        return True

    def load(self):
        pass

    def apply_obfuscation(self):
        pass

    def save(self, data):
        pass
