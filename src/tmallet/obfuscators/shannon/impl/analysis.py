from tmallet.shannon.calc import ShannonBERT
from tmallet.shannon.visualise import ShannonVisualiser
from typing import Union, List, Optional


class ShannonAnalyser:
    def __init__(self, device: str = "cuda"):
        self.calculator = ShannonBERT(device=device)
        self.visualiser = ShannonVisualiser()

    def get_distribution_by_word(
        self, texts: Union[List[str], str], plot_to: Optional[str] = None
    ):
        if isinstance(texts, List):
            processed_texts = [self.calculator.get_text_stats(t) for t in texts]
            mi = [
                [word.mutual_information for word in t.get_words()]
                for t in processed_texts
            ]
        elif isinstance(texts, str):
            processed_texts = self.calculator.get_text_stats(texts)
            mi = [word.mutual_information for word in processed_texts.get_words()]
        else:
            raise ValueError(f"Please provide a valid either a list or str for texts, not {type(texts)}")

        if plot_to:
            print(mi)
            self.visualiser.prepare_data(
                mutual_info=mi,
                flatten=True,
            )
            density = self.visualiser.plot_density(show_median=True)
            density.save(plot_to)
        return processed_texts
