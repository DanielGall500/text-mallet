from typing import Any

import spacy

_models = {}


def get_spacy_nlp(pipeline="ner") -> Any:
    spacy.prefer_gpu()
    if pipeline not in _models:
        if pipeline == "ner":
            _models[pipeline] = spacy.load("en_core_web_trf", disable=["parser"])
        elif pipeline == "lemma":
            _models[pipeline] = spacy.load(
                "en_core_web_trf", disable=["parser", "ner", "textcat"]
            )
        elif pipeline == "full":
            _models[pipeline] = spacy.load("en_core_web_trf")
        elif pipeline == "de":
            _models[pipeline] = spacy.load("de_dep_news_trf")
    return _models[pipeline]
