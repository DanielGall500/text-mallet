from tmallet.obfuscators.shannon.visualise import ShannonVisualiser
from wordfreq import get_frequency_dict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.tokenize import sent_tokenize
from typing import List, Optional
from dataclasses import dataclass
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn.functional as F
import nltk
import unicodedata

nltk.download("punkt_tab", quiet=True)

# == default model used for approximating Surprisal(word|context)

# ModernBERT (Warner et al., ACL 2025)
DEFAULT_MODEL_EN = "bert-base-cased"

# ModernGBERT 134M (Wunderle et al., 2025)
DEFAULT_MODEL_DE = "LSX-UniWue/ModernGBERT_134M"

freq_dict_en = get_frequency_dict("en", "best")
freqs_en = list(freq_dict_en.values())
freq_en_p5 = np.percentile(freqs_en, 5)

freq_dict_de = get_frequency_dict("de", "best")
freqs_de = list(freq_dict_de.values())
freq_de_p5 = np.percentile(freqs_de, 5)

freq_dict = {
    "en": {"dict": freq_dict_en, "p5": freq_en_p5},
    "de": {"dict": freq_dict_de, "p5": freq_de_p5},
}


@dataclass
class WordStat:
    word: str
    contextual_surprisal: float
    lang: str

    def __str__(self) -> str:
        return f"(w: {self.word}, I(w): {round(self.mutual_information,4)})"

    @property
    def contextual_probability(self):
        # P(X) = 2 ^ -S(X)
        return 2**-self.contextual_surprisal

    @property
    def mutual_information(self):
        # if punctuation, default to no information contributed
        if len(self.word) == 1 and (
            unicodedata.category(self.word).startswith("P")
            or unicodedata.category(self.word).startswith("S")
        ):
            return 0

        # find P(word) in lookup table
        # if not found, then try the same word but lowercase,
        # if not again, then it's likely names, dialectal spellings, etc,
        # give default 25th percentile freq (rare)
        prior_prob = freq_dict[self.lang]["dict"].get(self.word)
        if not prior_prob:
            prior_prob = freq_dict[self.lang]["dict"].get(
                self.word.lower(), freq_dict[self.lang]["p5"]
            )

        prior_surprisal = -math.log2(prior_prob)

        # Pointwise MI(X;Y) = S(X) - S(X|Y)
        PMI = prior_surprisal - self.contextual_surprisal

        # Positive PMI
        # it is common to clip all negative values where PMI
        # is an approximation
        PPMI = max(PMI, 0)
        return PPMI


@dataclass
class TextStat:
    text: str
    word_stats: List[WordStat]

    def get_words(self) -> List[WordStat]:
        return self.word_stats

    def get_word_labels(self) -> List[str]:
        return [w.word for w in self.word_stats]

    def get_mutual_infos(self) -> List[float]:
        return [w.mutual_information for w in self.word_stats]

    def __str__(self) -> str:
        return "\n".join([str(w) for w in self.word_stats])


class ShannonBERT:
    """
    A class for analysing text the way Claude Shannon did so quite enjoy.
    Allows for easy computation of surprisal and mutual information.

    Surprisal is defined as the negative log probability of a word.
    In this case, we're usually taking that to mean either P(word|context) or P(word).
    It can be useful for identifying rare words or constructions.

    Mutual information tells us how much 'information' the context tells us about the word.
    I(word; context) = Surprisal(word) - Surprisal(word|context).
    It is a bit more useful in this context. Take a dummy example where Surprisal(word) = 10 bits.
    If the context makes the word much more likely, I(word; context) = 10 - 0 = 10 bits of Mutual Information.
    If the context does nothing to help, I(word; context) = 10 bits - 10 bits = 0 bits of Mutual Information
    It measures the change in our uncertainty about a word occurring after we see the context.
    It is high for words which are rare but make sense in the context. These are often content words.

    Args:
        model_name: Name of the pretrained BERT model (default: 'bert-base-cased')
        device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
    """
    current_text_stat = None

    def __init__(
        self,
        lang: str,
        prefer_gpu: bool = False,
        max_context_length: int = 8192
    ):
        self.lang = lang
        self.max_context_length = max_context_length

        match lang:
            case "en":
                model_name = DEFAULT_MODEL_EN
            case "de":
                model_name = DEFAULT_MODEL_DE
            case _:
                raise ValueError("Please provide a valid language (`en` or `de`.")

        self.model_name = model_name

        if prefer_gpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def set_max_context_length(self, max_l: int):
        self.max_context_length = max_l

    def get_distribution_by_word(self, texts: List[str], plot_to: Optional[str] = None):
        all_mi_values = []

        for text in texts:
            mi_vals = self.calculate_mutual_info(text)
            all_mi_values.append(mi_vals)

        if plot_to:
            visualiser = ShannonVisualiser()
            visualiser.prepare_data(
                mutual_info=all_mi_values,
                flatten=True,
            )
            density = visualiser.plot_density(show_median=True)
            density.save(plot_to, width=10, height=5, dpi=400)
        return all_mi_values

    def calculate_mutual_info(self, text: str):
        text_stats = self.get_text_stats(text)
        return [word.mutual_information for word in text_stats.word_stats]

    def get_text_stats(self, text: str):
        all_words = []

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"][0].to(self.device)
        offsets = enc["offset_mapping"][0]
        word_ids = enc.word_ids()

        surprisals = torch.zeros(len(input_ids), device=self.device)
        with torch.no_grad():
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                masked = input_ids.clone()
                masked[i] = self.tokenizer.mask_token_id
                logits = self.model(masked.unsqueeze(0).to(self.device)).logits[0, i]
                log_probs = F.log_softmax(logits, dim=-1)
                surprisals[i] = -log_probs[input_ids[i]]

        word_surprisal = {}
        word_spans = {}
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            word_surprisal.setdefault(wid, 0.0)
            word_surprisal[wid] += surprisals[i].item()
            word_spans.setdefault(wid, [offsets[i][0], offsets[i][1]])
            word_spans[wid][0] = min(word_spans[wid][0], offsets[i][0])
            word_spans[wid][1] = max(word_spans[wid][1], offsets[i][1])

        words_in_text = [text[start:end] for wid, (start, end) in word_spans.items()]
        word_surp_in_text = [word_surprisal[wid] for wid in word_spans.keys()]

        if len(words_in_text) != len(word_surp_in_text):
            raise ValueError("Words does not match surprisal calculations.")

        for word, word_contextual_surprisal in zip(words_in_text, word_surp_in_text):
            all_words.append(WordStat(
                lang=self.lang,
                word=word,
                contextual_surprisal=word_contextual_surprisal,
            ))

        self.current_text_stat = TextStat(text=text, word_stats=all_words)
        return self.current_text_stat

    def get_current_text_stat(self):
        return self.current_text_stat

    """
    def get_text_stats(self, text: str):
        text_by_sent = sent_tokenize(text)

        # stores results for all words in the entire text, though
        # individual MI scores are computed at the sentence level.
        all_words = []
        for sentence in text_by_sent:
            enc = self.tokenizer(
                sentence,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            input_ids = enc["input_ids"][0].to(self.device)
            offsets = enc["offset_mapping"][0]
            word_ids = enc.word_ids()

            # compute surprisals for each word
            surprisals = torch.zeros(len(input_ids), device=self.device)
            with torch.no_grad():
                for i, wid in enumerate(word_ids):
                    # ignore special tokens
                    if wid is None:
                        continue

                    # list of token IDs from the input
                    masked = input_ids.clone()

                    # set one to [MASK]
                    masked[i] = self.tokenizer.mask_token_id

                    # get logits for the word ID
                    logits = self.model(masked.unsqueeze(0).to(self.device)).logits[
                        0, i
                    ]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # get the -logp (i.e. surprisal) for token ID
                    # found in input ids at ith element
                    surprisals[i] = -log_probs[input_ids[i]]

            word_surprisal = {}
            word_spans = {}

            for i, wid in enumerate(word_ids):
                # again, ignore special tokens
                if wid is None:
                    continue

                # wid is NOT the ID in a tokeniser, it's an index assigned to
                # each token that maps it to a specific word.
                # e.g. [0,0,1,1,1,2] -> Six tokens but three words
                word_surprisal.setdefault(wid, 0.0)
                word_surprisal[wid] += surprisals[i].item()

                # word span sets word ID : (start, end)
                # gets rid of ## for instance in tokeniser
                word_spans.setdefault(wid, [offsets[i][0], offsets[i][1]])
                word_spans[wid][0] = min(word_spans[wid][0], offsets[i][0])
                word_spans[wid][1] = max(word_spans[wid][1], offsets[i][1])

            words_in_sent = [
                sentence[start:end] for wid, (start, end) in word_spans.items()
            ]
            word_surp_in_sent = [word_surprisal[wid] for wid in word_spans.keys()]

            if len(words_in_sent) == len(word_surp_in_sent):
                for word, word_contextual_surprisal in zip(
                    words_in_sent, word_surp_in_sent
                ):
                    word_stat = WordStat(
                        lang=self.lang,
                        word=word,
                        contextual_surprisal=word_contextual_surprisal,
                    )
                    all_words.append(word_stat)
            else:
                raise ValueError("Words does not match surprisal calculations.")

        return TextStat(text=text, word_stats=all_words)
    """

    def get_text_stats_batch(
        self,
        texts: List[str],
    ) -> List[TextStat]:
        """
        Calculate surprisal for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries, one per input text
        """
        results = []

        iterator = texts
        try:
            iterator = tqdm(texts, desc="Calculating surprisal...")
        except ImportError:
            pass

        for text in iterator:
            result = self.get_text_stats(
                text,
            )
            results.append(result)

        return results

    def get_text_stats_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
    ) -> Dataset:
        """
        Calculate surprisal for all examples in a HuggingFace dataset.

        Args:
            dataset: HuggingFace Dataset object
            text_column: Name of the column containing text

        Returns:
            Dataset with added columns for surprisals
        """

        def process_example(example):
            text = example[text_column]
            result = self.get_text_stats(
                text,
            )
            surprisal_column = {
                "surprisals": [word.contextual_surprisal for word in result]
            }
            return surprisal_column

        processed = dataset.map(process_example, desc="Calculating surprisal")

        return processed
