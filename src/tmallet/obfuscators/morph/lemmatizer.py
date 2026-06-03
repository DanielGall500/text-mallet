from typing import Dict

from spacy.tokens import Doc

from tmallet.obfuscators.base import SpaCyObfuscator


class LemmaObfuscator(SpaCyObfuscator):
    def obfuscate(self, doc: Doc, config: Dict = {}) -> str:
        return self._lemmatise(doc)

    def set_config(self, config: dict):
        pass

    def _lemmatise(self, doc: Doc) -> str:
        lemmatised_text = "".join([token.lemma_ + token.whitespace_ for token in doc])
        return lemmatised_text
