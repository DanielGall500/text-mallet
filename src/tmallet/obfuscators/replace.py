from tmallet.obfuscators.base import SpaCyObfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Literal, Dict
from spacy.tokens import Doc


class ReplaceObfuscator(SpaCyObfuscator):
    POS = Literal["NOUN", "PROPN"]
    Algorithm = Literal[
        "nouns-only", "nouns-and-prop-only", "no-nouns", "no-nouns-or-prop"
    ]

    def __init__(self):
        self.detok = TreebankWordDetokenizer()

    def obfuscate(
        self,
        doc: Doc,
        config: Dict = {"algorithm": "noun", "replace_with_pos": True},
    ) -> str:
        if "algorithm" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'algorithm' parameter."
            )

        algorithm = config["algorithm"]
        replace_with_pos = config["replace_with_pos"]

        match algorithm:
            case "noun":
                return self._nouns_only(doc, replace_with_pos)
            case "noun-propn":
                return self._nouns_and_prop_only(doc, replace_with_pos)
            case "no-noun":
                return self._no_nouns(doc, replace_with_pos)
            case "no-noun-propn":
                return self._no_nouns_or_propn(doc, replace_with_pos)
            case _:
                raise ValueError("Please provide a valid algorithm.")

    def _nouns_only(self, doc: Doc, replace_with_pos: bool = False) -> str:
        return self._keep_only(doc, ["NOUN"], replace_with_pos)

    def _nouns_and_prop_only(self, doc: Doc, replace_with_pos: bool = False) -> str:
        return self._keep_only(doc, ["NOUN", "PROPN"], replace_with_pos)

    def _no_nouns(self, doc: Doc, replace_with_pos: bool = False) -> str:
        return self._keep_all_except(doc, ["NOUN"], replace_with_pos)

    def _no_nouns_or_propn(self, doc: Doc, replace_with_pos: bool = False) -> str:
        return self._keep_all_except(doc, ["NOUN", "PROPN"], replace_with_pos)

    def _keep_only(
        self, doc: Doc, pos_tags: list[POS], replace_with_pos: bool = False
    ) -> str:
        remaining_tokens = []
        for token in doc:
            is_valid_pos = token.pos_ in pos_tags
            # if the POS is allowed to be kept
            if is_valid_pos:
                remaining_tokens.append(token.text)
            # if it's not allowed to be kept but is to be replaced
            elif replace_with_pos:
                remaining_tokens.append(token.pos_)

        if len(remaining_tokens) == 0:
            print("INVALID No remaining tokens for: ", doc)
        return self.detok.detokenize(remaining_tokens)

    def _keep_all_except(
        self, doc: Doc, pos_tags: list[POS], replace_with_pos: bool = False
    ) -> str:
        remaining_tokens = []
        for token in doc:
            is_to_be_removed = token.pos_ in pos_tags
            if not is_to_be_removed:
                remaining_tokens.append(token.text)
            elif replace_with_pos:
                remaining_tokens.append(token.pos_)
        return self.detok.detokenize(remaining_tokens)
