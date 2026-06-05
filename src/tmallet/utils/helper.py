import syntok.segmenter as segmenter

from tmallet.obfuscators.replacement_token import (
    DEFAULT_TOKEN,
    ReplacementMechanism,
)


def sent_tokenize(text) -> list[str]:
    # Get string as list of sentences
    return [
        "".join(token.spacing + token.value for token in sentence).lstrip()
        for paragraph in segmenter.analyze(text)
        for sentence in paragraph
    ]


# -- Flatten a Dictionary --
# function that turns a nested dictionary
# into an unnested dictionary with nested
# keys concatenated with each other
def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            # note that we return each item in a list rather than the item itself,
            # this is because we want to flatten dicts for batch processing,
            # which requires a list or other collection
            items[new_key] = v  # changed from list
    return items


def apply_obfuscation(output_dict, word_text, rm, mechanism_tok, condition, bound):
    if condition:
        output_dict[bound][rm].append(word_text)
    elif rm == "default" or rm == "POS":
        output_dict[bound][rm].append(mechanism_tok)
    else:
        # word is deleted, do nothing
        pass


def get_replacement_mechanism(
    mechanism: ReplacementMechanism, word_index, pos_tags=None
):
    match mechanism:
        case ReplacementMechanism.POS:
            if pos_tags:
                replacement_tok = pos_tags[word_index]
            else:
                raise ValueError(
                    "Cannot pass pos_tags as None for replacement mechanism."
                )
        case ReplacementMechanism.Default:
            replacement_tok = DEFAULT_TOKEN
        case ReplacementMechanism.Delete:
            replacement_tok = ""
        case _:
            raise ValueError(
                f"Please provide a valid replacement mechanism (provided {mechanism})."
            )
    return replacement_tok
