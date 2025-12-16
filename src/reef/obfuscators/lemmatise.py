from reef.obfuscators.base import Obfuscator


class LemmaObfuscator(Obfuscator):
    def obfuscate(self, text: str) -> str:
        return self._lemmatise(text)

    def _lemmatise(self, doc) -> str:
        lemmatised_text = "".join([token.lemma_ + token.whitespace_ for token in doc])
        return lemmatised_text
