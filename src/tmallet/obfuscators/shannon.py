from tmallet.obfuscators.base import Obfuscator
from tmallet.surprisal.calc import ShannonBERT
from typing import Dict

DEFAULT_MODEL = "bert-base-cased"


class ShannonObfuscator(Obfuscator):
    def __init__(self, device: str = "cpu"):
        self.shannon = ShannonBERT(model_name=DEFAULT_MODEL, device=device)

    def obfuscate(self, text: str, config: Dict = {}) -> str:
        mutual_info = self.shannon.calculate_mutual_info(text)
        print(mutual_info)
        return text
