import pytest

from tmallet import TMallet

SAMPLE_TEXT_EN = (
    "Printing is being used to make metal parts for aircraft and space vehicles"
)
SAMPLE_TEXT_DE = "Der Druck wird zur Herstellung von Metallteilen für Flugzeuge und Raumfahrzeuge eingesetzt"


def make_tmallet(lang: str, algorithm: str, config: dict):
    return TMallet(lang, prefer_gpu=False).load_obfuscator(
        algorithm=algorithm, config=config
    )


@pytest.mark.parametrize(
    "algorithm,strength",
    [
        ("scramble-hier", "weak"),
        ("scramble-hier", "weak"),
        ("scramble-hier", "strong"),
        ("scramble-hier", "strong"),
    ],
)
def test_structural_hier_returns_str_en(algorithm, strength):
    config = {"strength": strength}
    tmallet_en = make_tmallet("en", algorithm, config)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "algorithm,strength",
    [
        ("scramble-hier", "weak"),
        ("scramble-hier", "weak"),
        ("scramble-hier", "strong"),
        ("scramble-hier", "strong"),
    ],
)
def test_structural_preserves_words_en(algorithm, strength):
    config = {"strength": strength}
    tmallet_en = make_tmallet("en", algorithm, config)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert set(SAMPLE_TEXT_EN.lower().split()) == set(result.lower().split())


@pytest.mark.parametrize(
    "algorithm,strength",
    [
        ("scramble-hier", "weak"),
        ("scramble-hier", "weak"),
        ("scramble-hier", "strong"),
        ("scramble-hier", "strong"),
    ],
)
def test_structural_returns_str_de(algorithm, strength):
    config = {"strength": strength}
    tmallet_de = make_tmallet("de", algorithm, config)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "algorithm,strength",
    [
        ("scramble-hier", "weak"),
        ("scramble-hier", "weak"),
        ("scramble-hier", "strong"),
        ("scramble-hier", "strong"),
    ],
)
def test_structural_preserves_words_de(algorithm, strength):
    config = {"strength": strength}
    tmallet_de = make_tmallet("de", algorithm, config)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    print(result)
    assert set(SAMPLE_TEXT_DE.lower().split()) == set(result.lower().split())
