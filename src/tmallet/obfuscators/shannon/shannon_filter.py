from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.shannon.shannon_bert import ShannonBERT
from tmallet.obfuscators.shannon.visualise import ShannonVisualiser
from tmallet.utils.helper import get_replacement_mechanism, apply_obfuscation
from tmallet.obfuscators.shannon.config import (
    ShannonFilterConfig,
)
from nltk.tokenize.treebank import TreebankWordDetokenizer
import logging

from tmallet.utils.spacy_registry import LangConfig, SpaCyInterface

logging.getLogger("transformers").setLevel(logging.ERROR)


class ShannonFilter(Obfuscator):
    """
    Removes tokens depending on their Mutual Information, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """

    def __init__(
        self,
        lang: LangConfig,
        spacy_interface: SpaCyInterface,
        prefer_gpu: bool = False,
    ):
        self.shannon = ShannonBERT(lang=lang, prefer_gpu=prefer_gpu)
        self.detok = TreebankWordDetokenizer()
        self.spacy_interface = spacy_interface

    def visualise(self):
        shannon_visualiser = ShannonVisualiser()
        current_text_stat = self.shannon.get_current_text_stat()
        mi_vals = current_text_stat.get_mutual_infos()
        labels = current_text_stat.get_word_labels()
        vis_html = shannon_visualiser.display_sentence_heatmap(labels, mi_vals)
        return vis_html

    def set_config(self, config: ShannonFilterConfig):
        # check if multiple thresholds were specified
        self.threshold = (
            config.threshold
            if isinstance(config.threshold, list)
            else [config.threshold]
        )

        # check if multiple replacement mechanisms were specified
        self.replacement_mechanism = (
            config.replacement_mechanism
            if isinstance(config.replacement_mechanism, list)
            else [config.replacement_mechanism]
        )

        self.bounds = config.bound
        if isinstance(self.bounds, list):
            self.as_upper_bound = "upper" in self.bounds
            self.as_lower_bound = "lower" in self.bounds
        else:
            self.as_upper_bound = self.bounds == "upper"
            self.as_lower_bound = self.bounds == "lower"

        self.output_mi_values = config.output_mi_values
        self.max_context_length = config.max_context_length
        self.shannon.set_max_context_length(self.max_context_length)

        self.uses_pos_tagger = "POS" in self.replacement_mechanism

    def obfuscate(self, text: str) -> dict:
        shannon_stats_text = self.shannon.get_text_stats(text)
        word_labels = shannon_stats_text.get_word_labels()
        mi_values = shannon_stats_text.get_mutual_infos()

        # check if we need to parse the text for POS tags
        pos_tags = None
        if self.uses_pos_tagger:
            # note: this is the only case where the spacy interface
            # is used by the obfuscation class itself and
            # not handled by the pipeline
            pos_tags = self.spacy_interface.get_pos_tags_for_tokens(word_labels)

        reconstructed = {}
        for thresh in self.threshold:
            resulting_output = {
                "upper": {rm: [] for rm in self.replacement_mechanism},
                "lower": {rm: [] for rm in self.replacement_mechanism},
            }

            for i, word_text in enumerate(word_labels):
                # compute replacement tokens once per position
                mechanism_tokens = {
                    rm: get_replacement_mechanism(rm, i, pos_tags)
                    for rm in self.replacement_mechanism
                }

                for rm in self.replacement_mechanism:
                    mechanism_tok = mechanism_tokens[rm]
                    is_below_threshold = mi_values[i] < thresh

                    apply_obfuscation(
                        resulting_output,
                        word_text,
                        rm,
                        mechanism_tok,
                        self.as_upper_bound and is_below_threshold,
                        "upper",
                    )
                    apply_obfuscation(
                        resulting_output,
                        word_text,
                        rm,
                        mechanism_tok,
                        self.as_lower_bound and not is_below_threshold,
                        "lower",
                    )

            if self.as_lower_bound and self.as_upper_bound:
                reconstructed[thresh] = {}
                for bound in self.bounds:
                    reconstructed[thresh][bound] = {
                        rm: self.detok.detokenize(resulting_output[bound][rm])
                        for rm in self.replacement_mechanism
                    }
            else:
                active_bound = "upper" if self.as_upper_bound else "lower"
                reconstructed[thresh] = {
                    rm: self.detok.detokenize(resulting_output[active_bound][rm])
                    for rm in self.replacement_mechanism
                }

            final_output = {"mi": reconstructed}
            if self.output_mi_values:
                final_output["mi_values"] = mi_values

        return final_output
