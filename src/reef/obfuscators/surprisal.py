from reef.obfuscators.base import Obfuscator


class SurprisalObfuscator(Obfuscator):
    """
    Removes tokens depending on their surprisal scores, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """
    
    def obfuscate(self, text: str) -> str:
        return ""
