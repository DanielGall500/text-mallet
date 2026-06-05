from typing import List

from pydantic import BaseModel, ConfigDict


class ShannonFilterConfig(BaseModel):
    threshold: float | List[float] = 10
    bound: str | List[str] = "upper"
    replacement_mechanism: str | List[str] = "default"
    max_context_length: int = 128
    output_mi_values: bool = False
