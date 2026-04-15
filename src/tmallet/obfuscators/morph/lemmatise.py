from tmallet.obfuscators.base import SpaCyObfuscator
from spacy.tokens import Doc
from typing import Dict


class LemmaObfuscator(SpaCyObfuscator):
    def obfuscate(self, doc: Doc, config: Dict = {}) -> str:
        return self._lemmatise(doc)

    def _lemmatise(self, doc: Doc) -> str:
        lemmatised_text = "".join([token.lemma_ + token.whitespace_ for token in doc])
        return lemmatised_text
