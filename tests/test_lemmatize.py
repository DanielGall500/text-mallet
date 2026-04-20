import pytest
from tmallet import TMallet

SAMPLE_TEXT_EN = "Three-dimensional printing is being used to make metal parts for aircraft and space vehicles."
SAMPLE_TEXT_DE = "Der 3D-Druck wird zur Herstellung von Metallteilen für Flugzeuge und Raumfahrzeuge eingesetzt."

CONFIG = {"algorithm": "lemmatize"}


def make_tmallet(lang: str):
    return TMallet(lang, prefer_gpu=False).load_obfuscator(CONFIG)


@pytest.fixture(scope="module")
def tmallet_en():
    return make_tmallet("en")


@pytest.fixture(scope="module")
def tmallet_de():
    return make_tmallet("de")


def test_lemmatize_returns_string_en(tmallet_en):
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert isinstance(result, str)


def test_lemmatize_nonempty_en(tmallet_en):
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert len(result.strip()) > 0


def test_lemmatize_returns_string_de(tmallet_de):
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert isinstance(result, str)


def test_lemmatize_nonempty_de(tmallet_de):
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert len(result.strip()) > 0
