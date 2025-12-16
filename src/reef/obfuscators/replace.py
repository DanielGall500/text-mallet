from reef.obfuscators.base import Obfuscator
from reef.obfuscators.spacy_registry import get_spacy_nlp


class ReplaceObfuscator(Obfuscator):

    def obfuscate(self, text: str, algorithm="nouns-only") -> str:
        return self._to_nouns(text)

    def _keep_only(self, text: str, pos_tags: list[str]) -> str:
        nlp = self.spacy_nlp("ner")
        doc = nlp(text)
        remaining_tokens = []
        for token in doc:
            is_valid_pos = token.pos_ in pos_tags
            if is_valid_pos:
                remaining_tokens.append(token.text)
        return " ".join(remaining_tokens)


    def _nouns_only(self, text: str) -> str:
        doc = nlp(text)
        nouns = []
        for token in doc:
            is_noun = token.pos_ == "NOUN" or token.pos_ == "PROPN"
            if is_noun:
                nouns.append(token.text)
        return " ".join(nouns)
