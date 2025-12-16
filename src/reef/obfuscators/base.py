from reef.obfuscators.spacy_registry import get_spacy_nlp

class Obfuscator:
    def spacy_nlp(self, spacy_type: str = "full"):
        if not hasattr(self, "_spacy_nlp"):
            self._spacy_nlp = get_spacy_nlp(spacy_type)
        return self._spacy_nlp

    def obfuscate(self, text):
        raise NotImplementedError("This obfuscator is not implemented.")
