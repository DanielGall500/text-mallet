import pytest
from tmallet import TMallet

SAMPLE_TEXT_EN = "Three-dimensional printing is being used to make metal parts for aircraft and space vehicles."
SAMPLE_TEXT_DE = "Der 3D-Druck wird zur Herstellung von Metallteilen für Flugzeuge und Raumfahrzeuge eingesetzt."


def make_tmallet(lang: str, algorithm: str, replacement_mechanism: str):
    return TMallet(lang, prefer_gpu=False).load_obfuscator(
        {"algorithm": algorithm, "replacement_mechanism": replacement_mechanism}
    )


@pytest.mark.parametrize(
    "algorithm,replacement_mechanism",
    [
        ("noun-retain", "DELETE"),
        ("noun-propn-retain", "DELETE"),
        ("noun-remove", "DELETE"),
        ("noun-propn-remove", "DELETE"),
        ("noun-retain", "DEFAULT"),
        ("noun-propn-retain", "DEFAULT"),
        ("noun-remove", "DEFAULT"),
        ("noun-propn-remove", "DEFAULT"),
        ("noun-retain", "POS"),
        ("noun-propn-retain", "POS"),
        ("noun-remove", "POS"),
        ("noun-propn-remove", "POS"),
    ],
)
def test_pos_filtering_returns_string_en(algorithm, replacement_mechanism):
    tmallet_en = make_tmallet("en", algorithm, replacement_mechanism)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert isinstance(result, str)


def test_pos_noun_remove_reduces_length_en():
    tmallet_en = make_tmallet("en", "noun-remove", "DELETE")
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert len(result) < len(SAMPLE_TEXT_EN)


@pytest.mark.parametrize(
    "algorithm,replacement_mechanism",
    [
        ("noun-retain", "DELETE"),
        ("noun-propn-retain", "DELETE"),
        ("noun-remove", "DELETE"),
        ("noun-propn-remove", "DELETE"),
        ("noun-retain", "DEFAULT"),
        ("noun-propn-retain", "DEFAULT"),
        ("noun-remove", "DEFAULT"),
        ("noun-propn-remove", "DEFAULT"),
        ("noun-retain", "POS"),
        ("noun-propn-retain", "POS"),
        ("noun-remove", "POS"),
        ("noun-propn-remove", "POS"),
    ],
)
def test_pos_filtering_returns_string_de(algorithm, replacement_mechanism):
    tmallet_de = make_tmallet("de", algorithm, replacement_mechanism)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert isinstance(result, str)


def test_pos_noun_remove_reduces_length_de():
    tmallet_de = make_tmallet("de", "noun-remove", "DELETE")
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert len(result) < len(SAMPLE_TEXT_DE)
