from tmallet.obfuscators.replace import ReplaceObfuscator
from tmallet.obfuscators.lemmatise import LemmaObfuscator
from tmallet.obfuscators.information_reduction import SurprisalObfuscator
from tmallet.obfuscators.spacy_registry import get_spacy_nlp
from tmallet.obfuscators.scramble import (
    LinearScrambleObfuscator,
    HierarchicalScrambleObfuscator,
)

__all__ = [
    "ReplaceObfuscator",
    "LemmaObfuscator",
    "SurprisalObfuscator",
    "get_spacy_nlp",
    "LinearScrambleObfuscator",
    "HierarchicalScrambleObfuscator",
]
