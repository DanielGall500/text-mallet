from tmallet.obfuscators.replacement_token import (
    ReplacementMechanism,
    get_replacement_tok,
)
from tmallet.obfuscators.base import SpaCyObfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Literal, Dict
from spacy.tokens import Doc

DEFAULT_CONFIG = {"algorithm": "noun", "replacement_mechanism": "DEFAULT"}


class POSFilter(SpaCyObfuscator):
    POS = Literal["NOUN", "PROPN", "VERB"]
    Algorithm = Literal[
        "noun-retain", "nouns-and-prop-retain", "noun-remove", "noun-propn-remove"
    ]

    def __init__(self):
        self.detok = TreebankWordDetokenizer()

    def obfuscate(
        self,
        doc: Doc,
        config: Dict = DEFAULT_CONFIG,
    ) -> str:
        if "algorithm" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'algorithm' parameter."
            )

        replacement_mechanism: ReplacementMechanism = config["replacement_mechanism"]
        algorithm: str = config["algorithm"]

        match algorithm:
            case "noun-retain":
                return self._nouns_only(doc, replacement_mechanism)
            case "noun-propn-retain":
                return self._nouns_and_prop_only(doc, replacement_mechanism)
            case "noun-remove":
                return self._no_nouns(doc, replacement_mechanism)
            case "noun-propn-remove":
                return self._no_nouns_or_propn(doc, replacement_mechanism)
            case _:
                raise ValueError("Please provide a valid algorithm.")

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
                replacement_tok = get_replacement_tok(replacement_mechanism, token.pos_)
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
                replacement_tok = get_replacement_tok(replacement_mechanism, token.pos_)
                remaining_tokens.append(replacement_tok)
        return self.detok.detokenize(remaining_tokens)
