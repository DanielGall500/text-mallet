from pydantic import BaseModel, ConfigDict
from typing import List


class ShannonFilterConfig(BaseModel):
    threshold: float | List[float] = 10
    bound: str | List[str] = "upper"
    replacement_mechanism: str | List[str] = "def"
    max_context_length: int = 8192
    output_mi_values: bool = False
