from sacremoses import MosesDetokenizer
from spacy.tokens import Doc

from tmallet.obfuscators.base import SpaCyObfuscator
from tmallet.obfuscators.pos.config import FilterType, POSFilterConfig
from tmallet.obfuscators.replacement_token import (
    DEFAULT_TOKEN,
    ReplacementMechanism,
)


class POSFilter(SpaCyObfuscator):
    """
    Performs obfuscation based on Part-of-Speech (POS) filtering rules.

    This class inherits from SpaCyObfuscator and provides methods
    to filter tokens based on whether their POS tag is in a specified
    list of tags. It supports retaining, removing, and replacing filtered
    tokens using different replacement mechanisms.
    """

    def __init__(self):
        """
        Initializes the POSFilter.

        Sets up the NLTK TreebankDetokenizer used for joining tokens back into a string.
        """
        self.detok = MosesDetokenizer()

    def set_config(self, config: POSFilterConfig):
        """
        Sets the configuration for the POS filter.

        This method takes a POSFilterConfig object and populates the necessary
        internal state for obfuscation, including filter types, replacement
        mechanisms, and specific POS tags to consider.
        """
        self.filter_type = (
            config.filter_type
            if isinstance(config.filter_type, list)
            else [config.filter_type]
        )
        self.replacement_mechanism = (
            config.replacement_mechanism
            if isinstance(config.replacement_mechanism, list)
            else [config.replacement_mechanism]
        )

        self.is_single_obfuscation = (
            len(self.filter_type) == 1 and len(self.replacement_mechanism) == 1
        )

        # multiple can be specified for POS tags
        # but it still remains a single obfuscation i.e. returns a single str
        self.pos_tags = (
            config.pos_tags if isinstance(config.pos_tags, list) else [config.pos_tags]
        )

    def obfuscate(self, doc: Doc) -> dict | str:
        """
        Performs the obfuscation on the given spaCy Doc object.

        Determines whether to run single or multi-filter obfuscation based on the
        configured settings and calls the appropriate internal method.

        Args:
            doc: The spaCy Doc object to obfuscate.

        Returns:
            str or dict: The obfuscated string or a dictionary containing results
                          if multiple filter combinations are applied.
        """
        return (
            self._obfuscate_single(doc)
            if self.is_single_obfuscation
            else self._obfuscate_multi(doc)
        )

    def _obfuscate_single(self, doc: Doc) -> str:
        """
        Handles obfuscation when a single filter type and a single replacement
        mechanism are configured.

        Args:
            doc: The spaCy Doc object to obfuscate.

        Returns:
            str: The obfuscated string.

        Raises:
            ValueError: If the configured filter type or replacement mechanism is invalid.
        """
        ft = FilterType(self.filter_type[0])
        mech = ReplacementMechanism(self.replacement_mechanism[0])
        match ft:
            case "retain":
                result = self._keep_only(doc, self.pos_tags, mech)
            case "remove":
                result = self._keep_all_except(doc, self.pos_tags, mech)
            case _:
                raise ValueError(f"Please provide a valid filter type: {ft}.")
        return result

    def _obfuscate_multi(self, doc: Doc) -> dict:
        """
        Handles obfuscation when multiple filter types or multiple replacement
        mechanisms are configured.

        This method iterates over all combinations of configured filters and
        mechanisms, returning a dictionary mapping the results.

        Args:
            doc: The spaCy Doc object to obfuscate.

        Returns:
            dict: A dictionary containing the results for all filter combinations.

        Raises:
            ValueError: If an invalid filter type is encountered.
        """
        results = {}
        for ft in self.filter_type:
            results[ft] = {}
            for mech in self.replacement_mechanism:
                match ft:
                    case "retain":
                        results[ft][mech] = self._keep_only(
                            doc, self.pos_tags, ReplacementMechanism(mech)
                        )
                    case "remove":
                        results[ft][mech] = self._keep_all_except(
                            doc, self.pos_tags, ReplacementMechanism(mech)
                        )
                    case _:
                        raise ValueError(f"Please provide a valid filter type: {ft}.")
        return {"pos-filter": results}

    def _keep_only(
        self,
        doc: Doc,
        pos_tags: list[str],
        replacement_mechanism: ReplacementMechanism = ReplacementMechanism.Default,
    ) -> str:
        """
        Keeps only tokens whose POS tag matches one of the specified tags.

        Tokens that do not match are replaced according to the given
        replacement_mechanism.

        Args:
            doc: The spaCy Doc object.
            pos_tags: A list of POS tags (str) that should be retained.
            replacement_mechanism: The mechanism to use for replacement.

        Returns:
            str: The detokenized string with only specified tokens retained.
        """
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
                    case ReplacementMechanism.POS:
                        replacement_tok = token.pos_
                    case ReplacementMechanism.Default:
                        replacement_tok = DEFAULT_TOKEN
                    case ReplacementMechanism.Delete:
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
        replacement_mechanism: ReplacementMechanism = ReplacementMechanism.Default,
    ) -> str:
        """
        Keeps all tokens except those whose POS tag matches one of the specified tags.

        Tokens marked for removal are replaced according to the given
        replacement_mechanism.

        Args:
            doc: The spaCy Doc object.
            pos_tags: A list of POS tags (str) that should be excluded (removed).
            replacement_mechanism: The mechanism to use for replacement.

        Returns:
            str: The detokenized string with specified tokens replaced or removed.
        """
        remaining_tokens = []
        for token in doc:
            is_to_be_removed = token.pos_ in pos_tags
            if not is_to_be_removed:
                remaining_tokens.append(token.text)
            else:
                # otherwise, find out whether it should be replaced with nothing,
                # a default character, or the POS tag
                match replacement_mechanism:
                    case ReplacementMechanism.POS:
                        replacement_tok = token.pos_
                    case ReplacementMechanism.Default:
                        replacement_tok = DEFAULT_TOKEN
                    case ReplacementMechanism.Delete:
                        continue
                    case _:
                        raise ValueError(
                            f"Please provide a valid replacement mechanism (provided {replacement_mechanism})."
                        )

                remaining_tokens.append(replacement_tok)
                # otherwise, find out whether it should be replaced with nothing,
                # a default character, or the POS tag
        return self.detok.detokenize(remaining_tokens)
