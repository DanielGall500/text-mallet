import random

from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.structural.config import LinearScrambleConfig
from tmallet.utils import sent_tokenize


class LinearScrambleObfuscator(Obfuscator):
    """
    Obfuscates text by scrambling the order of words (Bag-of-Words approach)
    based on specified scrambling levels (sentence or document).
    Note: this method does not require any SpaCy or BERT models.
    """

    def set_config(self, config: LinearScrambleConfig):
        """
        Sets the configuration for the LinearScrambleObfuscator.

        The configuration dictates the scrambling level(s) (e.g., 'sentence', 'document')
        and the random seed used for scrambling.

        Args:
            config: The LinearScrambleConfig object containing the required settings.
        """
        random.seed(config.seed)
        self.level = config.level if isinstance(config.level, list) else [config.level]
        self.is_single_obfuscation: bool = len(self.level) == 1

    def obfuscate(
        self,
        text: str,
    ) -> dict | str:
        """
        Applies the obfuscation based on the configured scrambling levels.

        If only one level is configured, it returns the obfuscated string directly.
        If multiple levels are configured, it returns a dictionary containing the
        obfuscated text for each specified level.

        Args:
            text: The input string text to obfuscate.

        Returns:
            The obfuscated text (str) or a dictionary containing multiple
            obfuscated results (dict) depending on the previously set configuration.
        """
        if self.is_single_obfuscation:
            return self._obfuscate_single(text)
        else:
            return self._obfuscate_multi(text)

    def _obfuscate_single(
        self,
        text: str,
    ) -> str:
        lvl = self.level[0]
        match lvl:
            case "sentence":
                sentences = sent_tokenize(text)
                scrambled_sentences = [self._linear_scramble(s) for s in sentences]
                return " ".join(scrambled_sentences)
            case "document":
                return self._linear_scramble(text)
            case _:
                raise ValueError(f"Invalid scrambling level: {lvl}")

    def _obfuscate_multi(
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
