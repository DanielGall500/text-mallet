from pandas.core.series import algorithms
from reef.dataloaders.txt_loader import TxtLoader
from reef.obfuscators.replace import ReplaceObfuscator
from reef.obfuscators.lemmatise import LemmaObfuscator
from reef.obfuscators.scramble import (
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
)
from datasets import load_dataset
from reef.obfuscators.spacy_registry import get_spacy_nlp
import spacy


class ReefPipeline:
    def run(self, repo_name: str) -> None:
        dataset = load_dataset(repo_name)

        nlp = get_spacy_nlp("lemma")

        # Create obfuscator once
        lemma_obf = LemmaObfuscator()

        def lemmatise_batch(batch):
            texts = batch["text"]

            docs = nlp.pipe(
                texts,
                batch_size=2000,
            )

            batch["text_lemmas"] = [
                lemma_obf.obfuscate(doc) for doc in docs
            ]
            return batch

        transformed_dataset = dataset.map(
            lemmatise_batch,
            batched=True,
            batch_size=2000,
            desc="Lemmatising..."
        )

        transformed_dataset.save_to_disk("transformed_dataset")


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

        # issue: needs to be an element of randomness in the algorithms
        # for reconstructability to be more difficult.
        return True

    def load(self):
        pass

    def apply_obfuscation(self):
        pass

    def save(self, data):
        pass
