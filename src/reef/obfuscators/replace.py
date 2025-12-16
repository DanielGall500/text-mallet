from reef.obfuscators.base import Obfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from typing import Literal


class ReplaceObfuscator(Obfuscator):

    POS = Literal["NOUN", "PROPN"]
    Algorithm = Literal[
        "nouns-only", "nouns-and-prop-only", "no-nouns", "no-nouns-or-prop"
    ]

    def obfuscate(self, text: str, algorithm: Algorithm = "nouns-only") -> str:
        self.nlp = self.spacy_nlp("ner")
        if algorithm == "nouns-only":
            return self._nouns_only(text)
        elif algorithm == "nouns-and-prop-only":
            return self._nouns_only(text)
        elif algorithm == "no-nouns":
            return self._no_nouns(text)
        elif algorithm == "no-nouns-or-prop":
            return self._no_nouns_or_propn(text)

    def _nouns_only(self, text: str) -> str:
        return self._keep_only(text, ["NOUN"])

    def _nouns_and_prop_only(self, text: str) -> str:
        return self._keep_only(text, ["NOUN", "PROPN"])

    def _no_nouns(self, text: str) -> str:
        return self._keep_all_except(text, ["NOUN"])

    def _no_nouns_or_propn(self, text: str) -> str:
        return self._keep_all_except(text, ["NOUN", "PROPN"])

    def _keep_only(self, text: str, pos_tags: list[POS]) -> str:
        doc = self.nlp(text)
        remaining_tokens = []
        for token in doc:
            is_valid_pos = token.pos_ in pos_tags
            if is_valid_pos:
                remaining_tokens.append(token.text)
        return TreebankWordDetokenizer().detokenize(remaining_tokens)

    def _keep_all_except(self, text: str, pos_tags: list[POS]) -> str:
        doc = self.nlp(text)
        remaining_tokens = []
        for token in doc:
            is_to_be_removed = token.pos_ in pos_tags
            if not is_to_be_removed:
                remaining_tokens.append(token.text)
        return TreebankWordDetokenizer().detokenize(remaining_tokens)
