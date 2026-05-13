from tmallet.obfuscators.base import Obfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from typing import Dict
import random

DEFAULT_SEED = 100
DEFAULT_LINEAR_CONFIG = {"level": "document", "seed": DEFAULT_SEED}


# the linear scrambler does not use SpaCy
class LinearScrambleObfuscator(Obfuscator):
    def obfuscate(
        self,
        text: str,
        config: Dict = DEFAULT_LINEAR_CONFIG,
    ) -> dict:

        if "seed" not in config.keys():
            seed = DEFAULT_SEED
        else:
            seed = config["seed"]
        random.seed(seed)

        if "level" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'level' parameter to determine whether scrambling occurs at document or sentence level."
            )
        level = config["level"]

        # check if multiple replacement mechanisms were specified
        is_multiple_levels = isinstance(level, list)
        if not is_multiple_levels:
            level = [level]

        result = {}
        for l in level:
            match l:
                case "sentence":
                    sentences = sent_tokenize(text)
                    scrambled_sentences = [self._linear_scramble(s) for s in sentences]
                    join_scrambled_sentences = " ".join(scrambled_sentences)
                    result[l] = join_scrambled_sentences
                case "document":
                    result[l] = self._linear_scramble(text)
                case _:
                    raise ValueError(f"Invalid scrambling level: {level}")
        return { "scramble-linear": result }

    def _linear_scramble(self, text) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words
