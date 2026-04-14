from tmallet.obfuscators.base import Obfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from typing import Dict
import random

DEFAULT_LINEAR_CONFIG = {"level": "sentence", "seed": 100}
DEFAULT_SEED = 100


# the linear scrambler does not use SpaCy
class LinearScrambleObfuscator(Obfuscator):
    def obfuscate(
        self,
        text: str,
        config: Dict = DEFAULT_LINEAR_CONFIG,
    ) -> str:

        if "seed" not in config.keys():
            seed = DEFAULT_SEED
        else:
            seed = config["seed"]
        random.seed(seed)

        if "level" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'level' parameter to determine whether scrambling occurs at document or sentence level."
            )

        scramble_level = config["level"]

        if scramble_level == "sentence":
            sentences = sent_tokenize(text)
            scrambled_sentences = [self._linear_scramble(s) for s in sentences]
            return " ".join(scrambled_sentences)
        elif scramble_level == "document":
            return self._linear_scramble(text)
        else:
            raise ValueError(f"Invalid scrambling level: {scramble_level}")

    def _linear_scramble(self, text) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words
