from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.shannon.impl.calc import ShannonBERT
from tmallet.obfuscators.shannon.impl.analysis import ShannonAnalyser
from tmallet.obfuscators.replacement_token import (
    ReplacementMechanism,
    get_replacement_tok,
)
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
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": DEFAULT_REPLACEMENT_MECHANISM,
    "output_mi_values": True
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

    def obfuscate(self, text: str, config: Dict = DEFAULT_CONFIG) -> Dict[float, str]:
        if "threshold" not in config.keys():
            max_mutual_info = DEFAULT_THRESHOLD
        else:
            max_mutual_info = config["threshold"]

        if "replacement_mechanism" not in config.keys():
            replacement_mechanism = "DEFAULT"
        else:
            replacement_mechanism = config["replacement_mechanism"]

        if "as_upper_bound" not in config.keys():
            as_upper_bound = True
        else:
            as_upper_bound = config["as_upper_bound"]

        if "as_lower_bound" not in config.keys():
            as_lower_bound = True
        else:
            as_lower_bound = config["as_lower_bound"]

        if "output_mi_values" not in config.keys():
            output_mi_values = False
        else:
            output_mi_values = config["output_mi_values"]

        shannon_stats_text = self.shannon.get_text_stats(text)
        words = shannon_stats_text.get_words()
        word_labels = shannon_stats_text.get_word_labels()
        mi_values = shannon_stats_text.get_mutual_infos()

        if replacement_mechanism == "POS":
            pos_tags = get_pos_tags([w.word for w in words])
        else:
            replacement_tok = get_replacement_tok(replacement_mechanism, None)

        # check if multiple thresholds were specified
        is_multiple_thresholds = isinstance(max_mutual_info, list)
        if not is_multiple_thresholds:
            max_mutual_info = [max_mutual_info]

        reconstructed = {}
        for thresh in max_mutual_info:
            resulting_output_lower_bound = []
            resulting_output_upper_bound = []

            for i, word_text in enumerate(word_labels):
                # check if we need to update the replacement token,
                # which is only the case if we're using POS tags
                if replacement_mechanism == "POS":
                    replacement_tok = pos_tags[i]

                if as_upper_bound:
                    if mi_values[i] <= thresh:
                        resulting_output_upper_bound.append(word_text)
                    else:
                        resulting_output_upper_bound.append(replacement_tok)

                if as_lower_bound:
                    if mi_values[i] >= thresh:
                        resulting_output_lower_bound.append(word_text)
                    else:
                        resulting_output_lower_bound.append(replacement_tok)

            reconstructed[thresh] = {
                "as_upper_bound": (
                    self.detok.detokenize(resulting_output_upper_bound)
                    if as_upper_bound
                    else None
                ),
                "as_lower_bound": (
                    self.detok.detokenize(resulting_output_lower_bound)
                    if as_lower_bound
                    else None
                ),
            }

            if output_mi_values:
                reconstructed[thresh]["mi_values"] = mi_values

        return reconstructed
