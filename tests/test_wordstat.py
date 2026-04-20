import math
import string
import pytest
from tmallet.obfuscators.shannon.impl.calc import WordStat, FREQ_P25, freq_dict


def test_contextual_probability_basic():
    ws = WordStat(word="hello", contextual_surprisal=3.0)
    assert ws.contextual_probability == pytest.approx(2**-3.0)


def test_contextual_probability_zero_surprisal():
    ws = WordStat(word="hello", contextual_surprisal=0.0)
    assert ws.contextual_probability == pytest.approx(1.0)


def test_mutual_information_known_word():
    ws = WordStat(word="hello", contextual_surprisal=3.0)
    prior_prob = freq_dict.get("hello", 0)
    expected_mi = -math.log2(prior_prob) - 3.0
    assert ws.mutual_information == pytest.approx(expected_mi)


def test_mutual_information_punctuation_returns_zero():
    for ch in string.punctuation:
        ws = WordStat(word=ch, contextual_surprisal=2.0)
        print(ws.mutual_information)
        assert ws.mutual_information == 0


def test_mutual_information_unknown_word_uses_freq_p25():
    ws = WordStat(word="zzzzqqqq", contextual_surprisal=5.0)
    expected_mi = -math.log2(FREQ_P25) - 5.0
    assert ws.mutual_information == pytest.approx(expected_mi)


def test_str_representation():
    ws = WordStat(word="hello", contextual_surprisal=3.0)
    result = str(ws)
    assert "hello" in result
    assert str(round(ws.mutual_information, 4)) in result
