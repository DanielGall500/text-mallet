from typing import Literal

ReplacementMechanism = Literal["DELETE", "DEFAULT", "POS"]
DEFAULT_TOKEN = "_"


def get_replacement_tok(preference: ReplacementMechanism, pos_tag: str):
    match preference:
        case "DELETE":
            return ""
        case "DEFAULT":
            return DEFAULT_TOKEN
        case "POS":
            return pos_tag
        case _:
            raise ValueError(f"Replacement preference not valid: {preference}")
