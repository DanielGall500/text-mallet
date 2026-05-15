from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.shannon.impl.calc import ShannonBERT
from tmallet.obfuscators.shannon.impl.analysis import ShannonAnalyser
from tmallet.obfuscators.replacement_token import (
    ReplacementMechanism,
    get_replacement_tok,
    DEFAULT_TOKEN,
)
from typing import Dict, Union, List
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging

from tmallet.utils.spacy_registry import LangConfig, SpaCyInterface

logging.getLogger("transformers").setLevel(logging.ERROR)

# == default model used for approximating Surprisal(word|context)

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019)
DEFAULT_MODEL_EN = "bert-base-cased" 

# German’s Next Language Model (Chan et al., COLING 2020)
DEFAULT_MODEL_DE = "deepset/gbert-base" 

# == config parameters ==
DEFAULT_THRESHOLD = 10
DEFAULT_REPLACEMENT_MECHANISM = "DEFAULT"
DEFAULT_CONFIG = {
    "threshold": DEFAULT_THRESHOLD,
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": DEFAULT_REPLACEMENT_MECHANISM,
    "output_mi_values": True,
}

def get_replacement_mechanism(mechanism, word_index, pos_tags=None):
    match mechanism:
        case "POS":
            replacement_tok = pos_tags[word_index]
        case "DEFAULT":
            replacement_tok = DEFAULT_TOKEN
        case "DELETE":
            replacement_tok = None
        case _:
            raise ValueError(
                f"Please provide a valid replacement mechanism (provided {mechanism})."
            )
    return replacement_tok

class ShannonFilter(Obfuscator):
    """
    Removes tokens depending on their Mutual Information, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """

    def __init__(
        self, lang: LangConfig, spacy_interface: SpaCyInterface, device: str = "cpu"
    ):
        match lang:
            case "en":
                model_name = DEFAULT_MODEL_EN
            case "de":
                model_name = DEFAULT_MODEL_DE
            case _:
                raise ValueError("Please provide a valid language (`en` or `de`.")

        self.shannon = ShannonBERT(model_name=model_name, device=device)
        self.detok = TreebankWordDetokenizer()
        self.spacy_interface = spacy_interface

    # todo: improve function SOC
    def obfuscate(self, text: str, config: Dict = DEFAULT_CONFIG) -> dict:
        # set the relevant parameters
        max_mutual_info = config.get("threshold", DEFAULT_THRESHOLD)
        replacement_mechanism = config.get("replacement_mechanism", "DEFAULT")
        as_upper_bound = config.get("as_upper_bound", True)
        as_lower_bound = config.get("as_lower_bound", True)
        output_mi_values = config.get("output_mi_values", False)

        shannon_stats_text = self.shannon.get_text_stats(text)
        words = shannon_stats_text.get_words()
        word_labels = shannon_stats_text.get_word_labels()
        mi_values = shannon_stats_text.get_mutual_infos()

        # check if multiple replacement mechanisms were specified
        is_multiple_replacements = isinstance(replacement_mechanism, list)
        if not is_multiple_replacements:
            replacement_mechanism = [replacement_mechanism]

        # check if multiple thresholds were specified
        is_multiple_thresholds = isinstance(max_mutual_info, list)
        if not is_multiple_thresholds:
            max_mutual_info = [max_mutual_info]

        # check if we need to parse the text for POS tags
        pos_tags = None
        if "POS" in replacement_mechanism:
            # note: this is the only case where the spacy interface
            # is used by the obfuscation class itself and
            # not handled by the pipeline
            pos_tags = self.spacy_interface.get_pos_tags_for_tokens(word_labels)

        reconstructed = {}
        for thresh in max_mutual_info:
            resulting_output_lower_bound = {rm: [] for rm in replacement_mechanism}
            resulting_output_upper_bound = {rm: [] for rm in replacement_mechanism}

            for i, word_text in enumerate(word_labels):
                # compute replacement tokens once per position
                mechanism_tokens = {
                    rm: get_replacement_mechanism(rm, i, pos_tags)
                    for rm in replacement_mechanism
                }

                for rm in replacement_mechanism:
                    mechanism_tok = mechanism_tokens[rm]
                    is_below_threshold = mi_values[i] < thresh

                    if as_upper_bound:
                        if is_below_threshold:
                            resulting_output_upper_bound[rm].append(word_text)
                        elif rm == "DEFAULT" or rm == "POS":
                            resulting_output_upper_bound[rm].append(mechanism_tok)
                        else:
                            # word is deleted, do nothing
                            pass

                    if as_lower_bound:
                        if not is_below_threshold:
                            resulting_output_lower_bound[rm].append(word_text)
                        elif rm == "DEFAULT" or rm == "POS":
                            resulting_output_lower_bound[rm].append(mechanism_tok)
                        else:
                            # word is deleted, do nothing
                            pass

            reconstructed[thresh] = {
                "as_upper_bound": (
                    {rm: self.detok.detokenize(resulting_output_upper_bound[rm]) for rm in replacement_mechanism}
                    if as_upper_bound else None
                ),
                "as_lower_bound": (   
                    {rm: self.detok.detokenize(resulting_output_lower_bound[rm]) for rm in replacement_mechanism}
                    if as_lower_bound else None
                ),
            }
            if output_mi_values:
                reconstructed[thresh]["mi_values"] = mi_values

        return { "mi": reconstructed }
