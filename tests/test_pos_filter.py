import pytest

from tmallet import TMallet

SAMPLE_TEXT_EN = "Three-dimensional printing is being used to make metal parts for aircraft and space vehicles."
SAMPLE_TEXT_DE = "Der 3D-Druck wird zur Herstellung von Metallteilen für Flugzeuge und Raumfahrzeuge eingesetzt."
ALGORITHM = "pos-filter"


def make_tmallet(lang: str, algorithm: str, config: dict):
    return TMallet(lang, prefer_gpu=False).load_obfuscator(algorithm, config)


@pytest.mark.parametrize(
    "filter_type,pos_tags,replacement_mechanism",
    [
        ("retain", ["NOUN"], "delete"),
        ("retain", ["NOUN", "PROPN"], "delete"),
        ("remove", ["NOUN"], "delete"),
        ("remove", ["NOUN", "PROPN"], "delete"),
        ("retain", ["NOUN"], "default"),
        ("retain", ["NOUN", "PROPN"], "default"),
        ("remove", ["NOUN"], "default"),
        ("remove", ["NOUN", "PROPN"], "default"),
        ("retain", ["NOUN"], "POS"),
        ("retain", ["NOUN", "PROPN"], "POS"),
        ("remove", ["NOUN"], "POS"),
        ("remove", ["NOUN", "PROPN"], "POS"),
    ],
)
def test_pos_filtering_returns_str_en(filter_type, pos_tags, replacement_mechanism):
    config = {
        "filter_type": filter_type,
        "pos_tags": pos_tags,
        "replacement_mechanism": replacement_mechanism,
    }
    tmallet_en = make_tmallet("en", ALGORITHM, config)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "filter_type,pos_tags,replacement_mechanism",
    [
        ("retain", ["NOUN"], "delete"),
        ("retain", ["NOUN", "PROPN"], "delete"),
        ("remove", ["NOUN"], "delete"),
        ("remove", ["NOUN", "PROPN"], "delete"),
        ("retain", ["NOUN"], "default"),
        ("retain", ["NOUN", "PROPN"], "default"),
        ("remove", ["NOUN"], "default"),
        ("remove", ["NOUN", "PROPN"], "default"),
        ("retain", ["NOUN"], "POS"),
        ("retain", ["NOUN", "PROPN"], "POS"),
        ("remove", ["NOUN"], "POS"),
        ("remove", ["NOUN", "PROPN"], "POS"),
    ],
)
def test_pos_filtering_returns_str_de(filter_type, pos_tags, replacement_mechanism):
    config = {
        "filter_type": filter_type,
        "pos_tags": pos_tags,
        "replacement_mechanism": replacement_mechanism,
    }
    tmallet_en = make_tmallet("de", ALGORITHM, config)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_DE)
    assert isinstance(result, str)
