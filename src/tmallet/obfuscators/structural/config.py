from pydantic import BaseModel, ConfigDict
from typing import List

DEFAULT_SEED = 100

class LinearScrambleConfig(BaseModel):
    level: str | List[str] = "document"
    seed: int = DEFAULT_SEED
    
class HierarchicalScrambleConfig(BaseModel):
    strength: str | List[str] = "strong"
    seed: int = DEFAULT_SEED
