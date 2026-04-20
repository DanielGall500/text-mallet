from spacy.language import Language
from spacy.tokens import Doc
from typing import Literal
import spacy

LangConfig = Literal["en", "de"]
spacy_transformer_en = "en_core_web_trf"
spacy_transformer_de = "de_dep_news_trf"


class SpaCyInterface:

    def __init__(self, lang: LangConfig = "en", prefer_gpu: bool = False):
        match lang:
            case "en":
                self.model_name = spacy_transformer_en
            case "de":
                self.model_name = spacy_transformer_de
            case _:
                raise ValueError(
                    f"The language code provided ({lang}) is not supported. Supported languages are English (en) and German (de)."
                )

        if prefer_gpu:
            spacy.prefer_gpu()

        self._full_model = spacy.load(self.model_name)
        self._active_model = self._full_model

    def set_pipeline(self, pipeline: str) -> None:
        nlp = self._full_model

        # reset all pipes to enabled before reconfiguring
        for name in nlp.component_names:
            if nlp.has_pipe(name):
                try:
                    nlp.enable_pipe(name)
                except ValueError:
                    pass

        match pipeline:
            case "pos":
                for name in nlp.component_names:
                    if name not in [
                        "transformer",
                        "tagger",
                        "morphologizer",
                        "attribute_ruler",
                        "lemmatizer",
                    ]:
                        nlp.disable_pipe(name)
            case "ner":
                for name in nlp.component_names:
                    if name not in ["transformer", "ner"]:
                        nlp.disable_pipe(name)
            case "lemma":
                for name in ["parser", "ner", "textcat"]:
                    if nlp.has_pipe(name):
                        nlp.disable_pipe(name)
            case "full":
                pass  # all pipes already enabled after reset
            case _:
                raise ValueError(
                    "Please provide a valid pipeline, i.e. one of `ner`, `lemma`, or `full`."
                )

        self._active_model = nlp

    def process(self, text: str) -> Doc:
        return self._active_model(text)

    def get_pos_tags_for_tokens(self, word_list: list[str]):
        """
        Takes a list of tokens and returns a list of universal POS tags.
        """
        spaces = [True] * (len(word_list) - 1) + [False]
        doc = Doc(self._active_model.vocab, words=word_list, spaces=spaces)

        for name, proc in self._active_model.pipeline:
            doc = proc(doc)

        return [token.pos_ for token in doc]
