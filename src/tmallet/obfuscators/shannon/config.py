from pydantic import BaseModel, ConfigDict
from typing import List

# == default model used for approximating Surprisal(word|context)

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019)
DEFAULT_MODEL_EN = "answerdotai/ModernBERT-base"

# German’s Next Language Model (Chan et al., COLING 2020)
# DEFAULT_MODEL_DE = "deepset/gbert-base"
DEFAULT_MODEL_DE = "LSX-UniWue/ModernGBERT_134M"


class ShannonFilterConfig(BaseModel):
    threshold: float | List[float] = 10
    as_upper_bound: bool = True
    as_lower_bound: bool = False
    replacement_mechanism: str | List[str] = "def"
    output_mi_values: bool = False
