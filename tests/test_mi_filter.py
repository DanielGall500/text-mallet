import pytest

from tmallet import TMallet

SAMPLE_TEXT_EN = "Three-dimensional printing is being used to make metal parts for aircraft and space vehicles."
SAMPLE_TEXT_DE = "Der 3D-Druck wird zur Herstellung von Metallteilen für Flugzeuge und Raumfahrzeuge eingesetzt."
ALGORITHM = "shannon"


def make_tmallet(lang: str, replacement_mechanism: str):
    return TMallet(lang, prefer_gpu=False).load_obfuscator(
        ALGORITHM,
        {
            "replacement_mechanism": replacement_mechanism,
            "threshold": 8,
            "bound": ["upper", "lower"],
            "output_mi_values": True,
            "max_context_length": 8192,
        },
    )


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_returns_dict_en(replacement_mechanism):
    tmallet_en = make_tmallet("en", replacement_mechanism)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert isinstance(result, dict)


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_result_has_expected_keys_en(replacement_mechanism):
    tmallet_en = make_tmallet("en", replacement_mechanism)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert "mi_values" in result.keys()


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_mi_values_present_en(replacement_mechanism):
    tmallet_en = make_tmallet("en", replacement_mechanism)
    result = tmallet_en.obfuscate(SAMPLE_TEXT_EN)
    assert "mi_values" in result.keys()


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_returns_dict_de(replacement_mechanism):
    tmallet_de = make_tmallet("de", replacement_mechanism)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert isinstance(result, dict)


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_result_has_expected_keys_de(replacement_mechanism):
    tmallet_de = make_tmallet("de", replacement_mechanism)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert "mi_values" in result.keys()


@pytest.mark.parametrize("replacement_mechanism", ["delete", "default", "POS"])
def test_shannon_mi_values_present_de(replacement_mechanism):
    tmallet_de = make_tmallet("de", replacement_mechanism)
    result = tmallet_de.obfuscate(SAMPLE_TEXT_DE)
    assert "mi_values" in result.keys()
