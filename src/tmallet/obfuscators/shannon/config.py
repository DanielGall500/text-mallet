from pydantic import BaseModel, ConfigDict
from typing import List


class ShannonFilterConfig(BaseModel):
    threshold: float | List[float] = 10
    as_upper_bound: bool = True
    as_lower_bound: bool = False
    replacement_mechanism: str | List[str] = "def"
    max_context_length: int = 8192
    output_mi_values: bool = False
