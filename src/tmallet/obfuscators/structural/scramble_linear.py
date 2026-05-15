from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.structural.config import LinearScrambleConfig
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from typing import Dict
import random


# the linear scrambler does not use SpaCy
class LinearScrambleObfuscator(Obfuscator):
    def set_config(self, config: LinearScrambleConfig):
        random.seed(config.seed)
        self.level = config.level if isinstance(config.level, list) else [config.level]

    def obfuscate(
        self,
        text: str,
    ) -> dict:
        result = {}
        for lvl in self.level:
            match lvl:
                case "sentence":
                    sentences = sent_tokenize(text)
                    scrambled_sentences = [self._linear_scramble(s) for s in sentences]
                    join_scrambled_sentences = " ".join(scrambled_sentences)
                    result[lvl] = join_scrambled_sentences
                case "document":
                    result[lvl] = self._linear_scramble(text)
                case _:
                    raise ValueError(f"Invalid scrambling level: {lvl}")
        return {"scramble-linear": result}

    def _linear_scramble(self, text) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words
