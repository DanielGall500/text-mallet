from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.shannon.impl.calc import ShannonBERT
from tmallet.obfuscators.shannon.impl.analysis import ShannonAnalyser
from tmallet.obfuscators.replacement_token import ReplacementMechanism, get_replacement_tok
from tmallet.utils.pos_tagger import get_pos_tags
from typing import Dict, Union, List
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL = "bert-base-cased"
DEFAULT_THRESHOLD = 10
DEFAULT_REPLACEMENT_MECHANISM = "DEFAULT"
DEFAULT_CONFIG = {
    "threshold": DEFAULT_THRESHOLD,
    "replacement_mechanism": DEFAULT_REPLACEMENT_MECHANISM
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


class ShannonFilter(Obfuscator):
    """
    Removes tokens depending on their Mutual Information, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """

    def __init__(self, device: str = "cpu"):
        self.shannon = ShannonBERT(model_name=DEFAULT_MODEL, device=device)
        self.detok = TreebankWordDetokenizer()

    def obfuscate(self, text: str, config: Dict = DEFAULT_CONFIG) -> str|List[str]:
        if "threshold" not in config.keys():
            max_mutual_info = DEFAULT_THRESHOLD
        else:
            max_mutual_info = config["threshold"]

        if "replacement_mechanism" not in config.keys():
            replacement_mechanism = "DEFAULT"
        else:
            replacement_mechanism = config["replacement_mechanism"]

        shannon_stats_text = self.shannon.get_text_stats(text)
        words = shannon_stats_text.get_words()

        if replacement_mechanism == "POS":
            pos_tags = get_pos_tags(words)


        if isinstance(max_mutual_info, list):
            surviving_words = []

            for i,w in enumerate(words):
                word_text = w.word
                if (w.mutual_information < thresh):
                    surviving_words.append(word_text)
                else:
                    replacement_tok = get_replacement_tok(replacement_mechanism, pos_tags[i])
                    surviving_words.append(replacement_tok)

            surviving_words = [[
                w.word if (w.mutual_information < thresh) else obfuscatory_token
                for w in words
            ] for thresh in max_mutual_info]
            reconstructed = {thresh: self.detok.detokenize(x) for thresh,x in zip(max_mutual_info,surviving_words)}
            return reconstructed
        else:
            surviving_words = [
                w.word if (w.mutual_information < max_mutual_info) else obfuscatory_token
                for w in words
            ]
            reconstructed = self.detok.detokenize(surviving_words)
            return reconstructed
