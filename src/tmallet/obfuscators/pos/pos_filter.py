from os import replace
from tmallet.obfuscators.replacement_token import (
    ReplacementMechanism,
    get_replacement_tok,
    DEFAULT_TOKEN,
)
from tmallet.obfuscators.base import SpaCyObfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Literal, Dict, List
from spacy.tokens import Doc

DEFAULT_CONFIG = {"algorithm": "pos-filter", "filter_type": "noun-retain", "replacement_mechanism": "DEFAULT"}


class POSFilter(SpaCyObfuscator):
    POS = Literal["NOUN", "PROPN", "VERB"]
    FilterType = Literal[
        "noun-retain", "nouns-and-prop-retain", "noun-remove", "noun-propn-remove"
    ]

    def __init__(self):
        self.detok = TreebankWordDetokenizer()

    def obfuscate(
        self,
        doc: Doc,
        config: Dict = DEFAULT_CONFIG,
    ) -> str:
        if "filter_type" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'type' parameter for this algorithm."
            )
        filter_type = config["filter_type"]

        if "replacement_mechanism" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'replacement_mechanism' parameter for this algorithm."
            )
        replacement_mechanism: ReplacementMechanism | List[ReplacementMechanism]= config["replacement_mechanism"]

        is_multiple_filters = isinstance(filter_type, list)
        if not is_multiple_filters:
            filter_type = [filter_type]

        is_multiple_replacement_mechanisms = isinstance(replacement_mechanism, list)
        if not is_multiple_replacement_mechanisms:
            replacement_mechanism = [replacement_mechanism]

        results = {}
        for ft in filter_type:
            results[ft] = {}
            for mech in replacement_mechanism:
                match ft:
                    case "noun-retain":
                        results[ft][mech] = self._nouns_only(doc, mech)
                    case "noun-propn-retain":
                        results[ft][mech] = self._nouns_and_prop_only(doc, mech)
                    case "noun-remove":
                        results[ft][mech] = self._no_nouns(doc, mech)
                    case "noun-propn-remove":
                        results[ft][mech] = self._no_nouns_or_propn(doc, mech)
                    case _:
                        raise ValueError(f"Please provide a valid filter type: {ft}.")
        return { "pos-filter": results }

    def _nouns_only(
        self, doc: Doc, replacement_mechanism: ReplacementMechanism = False
    ) -> str:
        return self._keep_only(doc, ["NOUN"], replacement_mechanism)

    def _nouns_and_prop_only(
        self, doc: Doc, replacement_mechanism: ReplacementMechanism = False
    ) -> str:
        return self._keep_only(doc, ["NOUN", "PROPN"], replacement_mechanism)

    def _no_nouns(
        self, doc: Doc, replacement_mechanism: bool = ReplacementMechanism
    ) -> str:
        return self._keep_all_except(doc, ["NOUN"], replacement_mechanism)

    def _no_nouns_or_propn(
        self, doc: Doc, replacement_mechanism: ReplacementMechanism = False
    ) -> str:
        return self._keep_all_except(doc, ["NOUN", "PROPN"], replacement_mechanism)

    def _keep_only(
        self,
        doc: Doc,
        pos_tags: list[POS],
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
        pos_tags: list[POS],
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
