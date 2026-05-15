from os import replace
from tmallet.obfuscators.pos.config import POSFilterConfig, FilterType
from tmallet.obfuscators.replacement_token import (
    ReplacementMechanism,
    DEFAULT_TOKEN,
)
from tmallet.obfuscators.base import SpaCyObfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import List
from spacy.tokens import Doc

class POSFilter(SpaCyObfuscator):
    def __init__(self):
        self.detok = TreebankWordDetokenizer()

    def set_config(self, config: POSFilterConfig):
        self.filter_type = config.filter_type if isinstance(config.filter_type, list) else [config.filter_type]
        self.pos_tags = config.pos_tags if isinstance(config.pos_tags, list) else [config.pos_tags]
        self.replacement_mechanism = config.replacement_mechanism if isinstance(config.replacement_mechanism, list) else [config.replacement_mechanism]

    def obfuscate(
        self,
        doc: Doc,
    ) -> str:
        results = {}
        for ft in self.filter_type:
            results[ft] = {}
            for mech in self.replacement_mechanism:
                match ft:
                    case "retain":
                        results[ft][mech] = self._keep_only(doc, self.pos_tags, mech)
                    case "remove":
                        results[ft][mech] = self._keep_all_except(doc, self.pos_tags, mech)
                    case _:
                        raise ValueError(f"Please provide a valid filter type: {ft}.")
        return { "pos-filter": results }

    def _keep_only(
        self,
        doc: Doc,
        pos_tags: list[str],
        replacement_mechanism: ReplacementMechanism = "DEFAULT",
    ) -> str:
        remaining_tokens = []
        for token in doc:
            is_kept = token.pos_ in pos_tags
            # if the word is allowed to be kept, then append it,
            if is_kept:
                remaining_tokens.append(token.text)
            else:
                # otherwise, find out whether it should be replaced with nothing,
                # a default character, or the POS tag
                match replacement_mechanism:
                    case "POS":
                        replacement_tok = token.pos_
                    case "DEFAULT":
                        replacement_tok = DEFAULT_TOKEN
                    case "DELETE":
                        continue
                    case _:
                        raise ValueError(
                            f"Please provide a valid replacement mechanism (provided {replacement_mechanism})."
                        )

                remaining_tokens.append(replacement_tok)

        return self.detok.detokenize(remaining_tokens)

    def _keep_all_except(
        self,
        doc: Doc,
        pos_tags: list[str],
        replacement_mechanism: ReplacementMechanism = "DEFAULT",
    ) -> str:
        remaining_tokens = []
        for token in doc:
            is_to_be_removed = token.pos_ in pos_tags
            if not is_to_be_removed:
                remaining_tokens.append(token.text)
            else:
                # otherwise, find out whether it should be replaced with nothing,
                # a default character, or the POS tag
                match replacement_mechanism:
                    case "POS":
                        replacement_tok = token.pos_
                    case "DEFAULT":
                        replacement_tok = DEFAULT_TOKEN
                    case "DELETE":
                        continue
                    case _:
                        raise ValueError(
                            f"Please provide a valid replacement mechanism (provided {replacement_mechanism})."
                        )

                remaining_tokens.append(replacement_tok)
                # otherwise, find out whether it should be replaced with nothing,
                # a default character, or the POS tag
        return self.detok.detokenize(remaining_tokens)
