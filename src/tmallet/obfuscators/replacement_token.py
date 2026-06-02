from enum import Enum


class ReplacementMechanism(Enum):
    Delete = "delete"
    Default = "default"
    POS = "POS"


DEFAULT_TOKEN = "_"


def get_replacement_tok(preference: ReplacementMechanism, pos_tag: str):
    match preference:
        case "delete":
            return None
        case "default":
            return DEFAULT_TOKEN
        case "POS":
            return pos_tag
        case _:
            raise ValueError(f"Replacement preference not valid: {preference}")
