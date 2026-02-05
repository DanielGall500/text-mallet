from spacy.tokens import Doc


class SpaCyObfuscator:
    def obfuscate(self, doc: Doc):
        raise NotImplementedError("SpaCy obfuscator is not implemented.")


class Obfuscator:
    def obfuscate(self, text: str):
        raise NotImplementedError("General-purpose obfuscator is not implemented.")
