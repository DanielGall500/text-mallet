from pydantic import BaseModel, ConfigDict
from typing import List
from enum import Enum

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
    filter_type: FilterType | List[FilterType] = "retain"
    pos_tags: List[POSTag] = ["NOUN","PROPN"]
    replacement_mechanism: str | List[str] = "def"
