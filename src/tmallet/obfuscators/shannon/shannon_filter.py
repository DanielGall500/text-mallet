import logging

from sacremoses import MosesDetokenizer

from tmallet.obfuscators.base import Obfuscator
from tmallet.obfuscators.replacement_token import ReplacementMechanism
from tmallet.obfuscators.shannon.config import (
    ShannonFilterConfig,
)
from tmallet.obfuscators.shannon.shannon_bert import ShannonBERT
from tmallet.utils.helper import apply_obfuscation, get_replacement_mechanism
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
        self.detok = MosesDetokenizer()
        self.spacy_interface = spacy_interface

    def set_config(self, config: ShannonFilterConfig):
        """
        Sets the configuration parameters for the ShannonFilter.

        This method processes the configuration object to set internal states,
        handling cases where multiple thresholds, bounds, or replacement
        mechanisms might be provided as lists.
        """
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

        self.is_single_obfuscation: bool = (
            len(self.threshold) == 1
            and len(self.replacement_mechanism) == 1
            and len(self.bounds) == 1
        ) or (not self.output_mi_values)

    def obfuscate(self, text: str) -> dict | str:
        """
        Main obfuscation function for mutual information obfuscation of text.
        Returns a string if a single mode is used, or a dictionary if
        multiple modes (e.g., multiple thresholds or bounds) are configured.
        """
        return (
            self._obfuscate_single(text)
            if self.is_single_obfuscation
            else self._obfuscate_multi(text)
        )

    def _obfuscate_single(self, text: str) -> str:
        """
        Obfuscate using a configuration intended for a single output string.
        """
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

        thresh = self.threshold[0]
        mech = ReplacementMechanism(self.replacement_mechanism[0])
        result = []

        for i, word_text in enumerate(word_labels):
            # compute replacement tokens once per position
            mechanism_tok = get_replacement_mechanism(mech, i, pos_tags)

            for rm in self.replacement_mechanism:
                is_below_threshold = mi_values[i] < thresh

                # apply obfuscation
                if (self.as_upper_bound and is_below_threshold) or (
                    self.as_lower_bound and not is_below_threshold
                ):
                    result.append(word_text)
                elif rm == "default" or rm == "POS":
                    result.append(mechanism_tok)
                else:
                    # word is deleted, do nothing
                    pass

        result = self.detok.detokenize(result)

        return result

    def _obfuscate_multi(self, text: str) -> dict:
        """
        Obfuscate using a configuration intended for dict output.
        """
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
                    rm: get_replacement_mechanism(ReplacementMechanism(rm), i, pos_tags)
                    for rm in self.replacement_mechanism
                }
                is_below_threshold = mi_values[i] < thresh

                for rm in self.replacement_mechanism:
                    mechanism_tok = mechanism_tokens[rm]

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
                final_output["mi_values"] = [round(mi, 2) for mi in mi_values]

        return final_output
