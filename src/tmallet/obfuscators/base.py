from spacy.tokens import Doc


class SpaCyObfuscator:
    def set_config(self, config):
        raise NotImplementedError("Implement set_config for setting config details.")

    def obfuscate(self, doc: Doc):
        raise NotImplementedError("SpaCy obfuscator is not implemented.")


class Obfuscator:
    def set_config(self, config):
        raise NotImplementedError("Implement set_config for setting config details.")

    def obfuscate(self, text: str):
        raise NotImplementedError("General-purpose obfuscator is not implemented.")
