import spacy
from spacy.language import Language

_models = {}

spacy_transformer_en = "en_core_web_trf"
spacy_transformer_de = "de_dep_news_trf"


def get_spacy_nlp(pipeline="ner", prefer_gpu: bool = False) -> Language:
    if prefer_gpu:
        spacy.prefer_gpu()

    if pipeline not in _models:
        match pipeline:
            case "ner":
                _models[pipeline] = spacy.load(spacy_transformer_en, disable=["parser"])
            case "lemma":
                _models[pipeline] = spacy.load(
                    spacy_transformer_en, disable=["parser", "ner", "textcat"]
                )
            case "full":
                _models[pipeline] = spacy.load(spacy_transformer_en)
            case "de":
                _models[pipeline] = spacy.load(spacy_transformer_de)
            case _:
                raise ValueError("Please provide a valid pipeline.")
    return _models[pipeline]
