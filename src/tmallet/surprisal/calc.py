from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Union, Optional
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F
from datasets import Dataset
import numpy as np
import torch
import nltk

nltk.download("punkt_tab")


class SurprisalCalculator:
    """
    A class for calculating token-level surprisal values using BERT models.

    Surprisal is defined as the negative log probability of a token:
    surprisal = -log(P(token|context))

    Args:
        model_name: Name of the pretrained BERT model (default: 'bert-base-cased')
        device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        batch_size: Batch size for processing multiple examples (default: 8)
    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def calculate_surprisal(
        self, text: str, return_tokens: bool = True, skip_special_tokens: bool = True
    ) -> Dict[str, List]:
        """
        Calculate surprisal for all tokens in a single text.
        The text is first split into sentences and surprisal is calculated on the sentence-level.

        Args:
            text: Input text to analyze
            return_tokens: Whether to return the tokens alongside surprisals
            skip_special_tokens: Whether to skip [CLS] and [SEP] tokens

        Returns:
            Dictionary containing:
                - 'surprisals': List of surprisal values
                - 'tokens': List of tokens (if return_tokens=True)
                - 'token_ids': List of token IDs
        """
        text_by_sent = sent_tokenize(text)

        all_surprisals = []
        all_token_ids = []

        for sentence in text_by_sent:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            input_ids = inputs["input_ids"].to(self.device)

            start_idx = 1 if skip_special_tokens else 0
            end_idx = (
                input_ids.size(1) - 1 if skip_special_tokens else input_ids.size(1)
            )

            surprisals_by_sent = []

            with torch.no_grad():
                for i in range(start_idx, end_idx):
                    masked_input = input_ids.clone()
                    masked_input[0, i] = self.tokenizer.mask_token_id

                    outputs = self.model(masked_input)
                    logits = outputs.logits[0, i]

                    probs = torch.softmax(logits, dim=-1)
                    token_id = input_ids[0, i].item()
                    token_prob = probs[token_id]

                    surprisal = -torch.log(token_prob).item()
                    surprisals_by_sent.append(surprisal)
                print([round(s, 4) for s in surprisals_by_sent])

            all_surprisals.extend(surprisals_by_sent)
            all_token_ids.extend(input_ids[0, start_idx:end_idx].cpu().tolist())

        result = {"surprisals": all_surprisals, "token_ids": all_token_ids}

        if return_tokens:
            tokens = self.tokenizer.convert_ids_to_tokens(result["token_ids"])
            result["tokens"] = tokens

        return result

    def calculate_word_level_surprisal(self, text: str):
        text_by_sent = sent_tokenize(text)

        all_surprisals = []
        all_words = []

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
                    if wid is None:
                        continue
                    masked = input_ids.clone()
                    masked[i] = self.tokenizer.mask_token_id
                    logits = self.model(masked.unsqueeze(0)).logits[0, i]
                    log_probs = F.log_softmax(logits, dim=-1)
                    surprisals[i] = -log_probs[input_ids[i]]

            word_surprisal = {}
            word_spans = {}

            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
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
            # word_ids_in_sent = word_spans.keys()
            word_surp_in_sent = [word_surprisal[wid] for wid in word_spans.keys()]

            all_words.extend(words_in_sent)
            all_surprisals.extend(word_surp_in_sent)

        result = {"surprisals": all_surprisals, "words": all_words}
        return result

    def calculate_surprisal_batch(
        self,
        texts: List[str],
        return_tokens: bool = True,
        skip_special_tokens: bool = True,
        show_progress: bool = False,
    ) -> List[Dict[str, List]]:
        """
        Calculate surprisal for multiple texts.

        Args:
            texts: List of input texts
            return_tokens: Whether to return tokens alongside surprisals
            skip_special_tokens: Whether to skip [CLS] and [SEP] tokens
            show_progress: Whether to show progress bar (requires tqdm)

        Returns:
            List of dictionaries, one per input text
        """
        results = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(texts, desc="Calculating surprisal")
            except ImportError:
                pass

        for text in iterator:
            result = self.calculate_surprisal(
                text,
                return_tokens=return_tokens,
                skip_special_tokens=skip_special_tokens,
            )
            results.append(result)

        return results

    def calculate_surprisal_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        return_tokens: bool = True,
        skip_special_tokens: bool = True,
        show_progress: bool = True,
    ) -> Dataset:
        """
        Calculate surprisal for all examples in a HuggingFace dataset.

        Args:
            dataset: HuggingFace Dataset object
            text_column: Name of the column containing text
            return_tokens: Whether to include tokens in output
            skip_special_tokens: Whether to skip [CLS] and [SEP] tokens
            show_progress: Whether to show progress bar

        Returns:
            Dataset with added columns for surprisals (and optionally tokens)
        """

        def process_example(example):
            text = example[text_column]
            result = self.calculate_surprisal(
                text,
                return_tokens=return_tokens,
                skip_special_tokens=skip_special_tokens,
            )

            output = {"surprisals": result["surprisals"]}
            if return_tokens:
                output["tokens"] = result["tokens"]

            return output

        processed = dataset.map(
            process_example, desc="Calculating surprisal" if show_progress else None
        )

        return processed

    def get_average_surprisal(
        self,
        texts: Union[List[str], str],
        average_type: str = "mean",
        skip_special_tokens: bool = True,
    ) -> float:
        """
        Calculate mean surprisal for a text.

        Args:
            text: Input text
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Mean surprisal value
        """
        all_surprisals = []
        if type(texts) is str:
            result = self.calculate_surprisal(
                texts, return_tokens=False, skip_special_tokens=skip_special_tokens
            )
            all_surprisals = result["surprisals"]
            return np.mean(result["surprisals"])
        elif type(texts) is List[str]:
            for t in texts:
                surprisals = self.calculate_surprisal(
                    texts, return_tokens=False, skip_special_tokens=skip_special_tokens
                )["surprisals"]
                mean_text_surp = np.mean(surprisals)
                all_surprisals.append(mean_text_surp)

        else:
            raise ValueError("Please provide either a string or list of strings.")

        if average_type == "mean":
            return np.mean(all_surprisals)
        elif average_type == "median":
            return np.median(all_surprisals)
        else:
            raise ValueError(f"Invalid average type given: {average_type}")

    def get_total_surprisal(self, text: str, skip_special_tokens: bool = True) -> float:
        """
        Calculate total surprisal for a text (sum of all token surprisals).

        Args:
            text: Input text
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Total surprisal value
        """
        result = self.calculate_surprisal(
            text, return_tokens=False, skip_special_tokens=skip_special_tokens
        )
        return np.sum(result["surprisals"])
