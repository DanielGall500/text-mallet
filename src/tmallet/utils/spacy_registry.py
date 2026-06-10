from typing import Literal

import spacy
from spacy.tokens import Doc

LangConfig = Literal["en", "de"]
default_english_model = "core_web"
default_german_model_news = "core_news"
default_german_model_dep = "dep_news"

_PIPELINE_ALLOWLISTS: dict[str, set[str]] = {
    "pos": {
        "tok2vec",  # md/lg/sm models
        "transformer",  # trf models
        "tagger",
        "morphologizer",
        "attribute_ruler",
        "lemmatizer",
        "sentencizer",
    },
    "ner": {
        "tok2vec",
        "transformer",
        "ner",
    },
}


class SpaCyInterface:
    def __init__(
        self,
        lang: LangConfig = "en",
        prefer_gpu: bool = False,
        model_type: Literal["sm", "md", "lg", "trf"] = "lg",
    ):
        match lang:
            case "en":
                self.model_name = f"{lang}_{default_english_model}_{model_type}"
            case "de":
                if model_type == "trf":
                    self.model_name = f"{lang}_{default_german_model_dep}_{model_type}"
                else:
                    self.model_name = f"{lang}_{default_german_model_news}_{model_type}"
            case _:
                raise ValueError(
                    f"The language code provided ({lang}) is not supported. "
                    f"Supported languages are English (en) and German (de)."
                )
        if prefer_gpu:
            spacy.prefer_gpu()

        self._full_model = spacy.load(self.model_name)
        self._full_model.add_pipe("sentencizer")
        self._active_model = self._full_model

    def set_pipeline(self, pipeline: str) -> None:
        for name in self._full_model.component_names:
            if self._full_model.has_pipe(name):
                self._full_model.enable_pipe(name)

        match pipeline:
            case "pos" | "ner":
                allowlist = _PIPELINE_ALLOWLISTS[pipeline]
                for name in self._full_model.component_names:
                    if name not in allowlist:
                        self._full_model.disable_pipe(name)
            case "full":
                pass  # already fully re-enabled above
            case _:
                raise ValueError(
                    "Please provide a valid pipeline: 'ner', 'pos', or 'full'."
                )

        self._active_model = self._full_model

    def process(self, text: list[str] | str) -> list[Doc] | Doc:
        if isinstance(text, str):
            return self._active_model(text)
        else:
            return list(self._active_model.pipe(text))  # was self._full_model — bug fix

    def get_pos_tags_for_tokens(self, word_list: list[str]):
        """
        Takes a list of tokens and returns a list of universal POS tags.
        """
        spaces = [True] * (len(word_list) - 1) + [False]
        doc = Doc(self._active_model.vocab, words=word_list, spaces=spaces)
        for name, proc in self._active_model.pipeline:
            doc = proc(doc)
        return [token.pos_ for token in doc]
