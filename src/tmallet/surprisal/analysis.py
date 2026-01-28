from tmallet.surprisal.calc import SurprisalCalculator
from tmallet.surprisal.visualise import SurprisalVisualiser
from typing import Union, List, Optional


class SurprisalAnalyser:
    def __init__(self, device: str = "cpu"):
        self.calculator = SurprisalCalculator(device=device)
        self.visualiser = SurprisalVisualiser()

    def get_mean(self, texts: Union[List[str], str]):
        return self.calculator.get_average_surprisal(texts, average_type="mean")

    def get_median(self, texts: Union[List[str], str]):
        return self.calculator.get_average_surprisal(texts, average_type="median")

    def get_surprisal_by_token(self, texts: Union[List[str], str]):
        if type(texts) is List[str]:
            processed_texts = self.calculator.calculate_surprisal_batch(texts)
        elif type(texts) is str:
            processed_texts = self.calculator.calculate_surprisal_batch([texts])
        else:
            raise ValueError("Please provide a string or list of strings.")
        return processed_texts

    def get_surprisal_by_word(self, texts: Union[List[str], str]):
        if type(texts) is List[str]:
            processed_texts = None
        elif type(texts) is str:
            processed_texts = self.calculator.calculate_word_level_surprisal(texts)
        else:
            raise ValueError("Please provide a string or list of strings.")
        return processed_texts

    def get_distribution_surprisal_by_token(
        self, texts: Union[List[str], str], plot_to: Optional[str] = None
    ):
        processed_texts = self.calculator.calculate_surprisal(texts)

        if plot_to:
            surprisals = [text["surprisals"] for text in processed_texts]
            self.visualiser.prepare_data(
                surprisals=surprisals,
            )
            density = self.visualiser.plot_density(show_median=True)
            density.save(plot_to)
        return processed_texts

    def get_distribution_surprisal_by_word(
        self, texts: Union[List[str], str], plot_to: Optional[str] = None
    ):
        processed_texts = self.get_surprisal_by_word(texts)

        if plot_to:
            surprisals = [text["surprisals"] for text in processed_texts]
            self.visualiser.prepare_data(
                surprisals=surprisals,
            )
            density = self.visualiser.plot_density(show_median=True)
            density.save(plot_to)
        return processed_texts
