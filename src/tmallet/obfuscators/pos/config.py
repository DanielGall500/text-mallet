from enum import Enum
from typing import List

from pydantic import BaseModel


class FilterType(str, Enum):
    Retain = "retain"
    Remove = "remove"


class POSTag(str, Enum):
    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"
    SPACE = "SPACE"


class POSFilterConfig(BaseModel):
    filter_type: FilterType | List[FilterType] = FilterType.Retain
    pos_tags: List[POSTag] = [POSTag.NOUN, POSTag.PROPN]
    replacement_mechanism: str | List[str] = "default"
