from tmallet.obfuscators.replacement_token import (
    DEFAULT_TOKEN,
)


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
            items[new_key] = [v]
    return items


def apply_obfuscation(output_dict, word_text, rm, mechanism_tok, condition, bound):
    if condition:
        output_dict[bound][rm].append(word_text)
    elif rm == "DEFAULT" or rm == "POS":
        output_dict[bound][rm].append(mechanism_tok)
    else:
        # word is deleted, do nothing
        pass


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
