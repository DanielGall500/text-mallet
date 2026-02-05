from typing import List, Optional
from dataclasses import dataclass
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
from datasets import Dataset
import torch
import nltk
from wordfreq import word_frequency
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")
DEFAULT_LANG = "en"


@dataclass
class WordStat:
    word: str
    contextual_surprisal: float

    def __str__(self) -> str:
        return f"(w: {self.word}, I(w): {round(self.mutual_information,4)}"

    @property
    def contextual_probability(self):
        # P(X) = 2 ^ -S(X)
        return 2**-self.contextual_surprisal

    @property
    def mutual_information(self):
        try:
            prior_prob = word_frequency(self.word, DEFAULT_LANG)
            prior_surprisal = -math.log2(prior_prob)
        except ValueError:
            print("Couldn't compute MI: ", self.word)
            return 0

        # Pointwise I(X;Y) = S(X) - S(X|Y)
        MI = prior_surprisal - self.contextual_surprisal
        return MI


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

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def calculate_mutual_info(self, text: str):
        text_stats = self.get_text_stats(text)
        return [word.mutual_information for word in text_stats.word_stats]

    def get_text_stats(self, text: str):
        text_by_sent = sent_tokenize(text)

        for sentence in text_by_sent:
            enc = self.tokenizer(
                sentence, return_tensors="pt", return_offsets_mapping=True
            )

            input_ids = enc["input_ids"][0]
            offsets = enc["offset_mapping"][0]
            word_ids = enc.word_ids()

            # compute surprisals for each word
            surprisals = torch.zeros(len(input_ids))
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
                    logits = self.model(masked.unsqueeze(0)).logits[0, i]
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

            all_words = []
            if len(words_in_sent) == len(word_surp_in_sent):
                for word, word_contextual_surprisal in zip(
                    words_in_sent, word_surp_in_sent
                ):
                    word_stat = WordStat(
                        word=word, contextual_surprisal=word_contextual_surprisal
                    )
                    all_words.append(word_stat)
            else:
                raise ValueError("Words does not match surprisal calculations.")

        return TextStat(text=text, word_stats=all_words)

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
