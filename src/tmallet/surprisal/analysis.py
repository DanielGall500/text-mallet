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

    def get_distribution(
        self, texts: Union[List[str], str], save_to: Optional[str] = None
    ):
        if type(texts) is List[str]:
            processed_texts = self.calculator.calculate_surprisal_batch(texts)
        elif type(texts) is str:
            processed_texts = self.calculator.calculate_surprisal_batch([texts])
        else:
            raise ValueError("Please provide a string or list of strings.")

        surprisals = [text["surprisals"] for text in processed_texts]
        self.visualiser.prepare_data(
            surprisals=surprisals,
        )
        density = self.visualiser.plot_density(show_median=True)
        if save_to:
            density.save(save_to)
        return density
