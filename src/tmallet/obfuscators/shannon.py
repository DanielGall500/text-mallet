from tmallet.obfuscators.base import Obfuscator
from tmallet.shannon.calc import ShannonBERT
from tmallet.shannon.analysis import ShannonAnalyser
from typing import Dict, Union, List
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL = "bert-base-cased"
DEFAULT_THRESHOLD = 10
DEFAULT_OBFUSCATORY_TOKEN = "_"
DEFAULT_CONFIG = {
    "threshold": DEFAULT_THRESHOLD,
    "replace_with": DEFAULT_OBFUSCATORY_TOKEN,
}


def analyse(texts: Union[List[str], str], save_plot_to: str = "dist.png"):
    analyser = ShannonAnalyser()
    analyser.get_distribution(texts, save_to=save_plot_to)
    mean_surp = analyser.get_mean(texts)
    median_surp = analyser.get_median(texts)

    print("====")
    print("Shannon Analysis")
    print(f"Mean: {mean_surp}")
    print(f"Median: {median_surp}")
    print(f"Distribution saved to: {save_plot_to}")
    print("====")


class ShannonObfuscator(Obfuscator):
    """
    Removes tokens depending on their Mutual Information, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """

    def __init__(self, device: str = "cpu"):
        self.shannon = ShannonBERT(model_name=DEFAULT_MODEL, device=device)
        self.detok = TreebankWordDetokenizer()

    def obfuscate(self, text: str, config: Dict = DEFAULT_CONFIG) -> str:
        if "threshold" not in config.keys():
            max_mutual_info = DEFAULT_THRESHOLD
        else:
            max_mutual_info = config["threshold"]

        if "threshold" not in config.keys():
            obfuscatory_token = DEFAULT_OBFUSCATORY_TOKEN
        else:
            obfuscatory_token = config["replace_with"]

        shannon_stats_text = self.shannon.get_text_stats(text)
        words = shannon_stats_text.get_words()
        surviving_words = [
            w.word if (w.mutual_information < max_mutual_info) else obfuscatory_token
            for w in words
        ]

        reconstructed = self.detok.detokenize(surviving_words)
        return reconstructed
