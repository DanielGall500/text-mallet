from tmallet.obfuscators.base import Obfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from typing import Dict
import random

DEFAULT_LINEAR_CONFIG = {"scramble_within_sentence": False, "seed": 100}
DEFAULT_SEED = 100


# the linear scrambler does not use SpaCy
class LinearScrambleObfuscator(Obfuscator):
    def obfuscate(
        self,
        text: str,
        config: Dict = DEFAULT_LINEAR_CONFIG,
    ) -> str:

        if "scramble_within_sentence" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'algorithm' parameter."
            )
        scramble_within_sentence = config["scramble_within_sentence"]

        if "seed" not in config.keys():
            seed = DEFAULT_SEED
        else:
            seed = config["seed"]
        random.seed(seed)

        if scramble_within_sentence:
            sentences = sent_tokenize(text)
            scrambled_sentences = [self._linear_scramble(s) for s in sentences]
            return " ".join(scrambled_sentences)
        else:
            return self._linear_scramble(text)

    def _linear_scramble(self, text) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words
