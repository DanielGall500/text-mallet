from tmallet.obfuscators.base import Obfuscator
from tmallet.surprisal import SurprisalAnalyser
from typing import Union, List, Dict


class SurprisalObfuscator(Obfuscator):
    """
    Removes tokens depending on their surprisal scores, as assigned
    by an NLU model from Hugging Face e.g. bert-base-cased.

    Recommended to run this using at least a GPU.
    """

    def __init__(self, replace_with: str = "_"):
        self.analyser = SurprisalAnalyser()
        self.obfuscation_replacement = replace_with

    def obfuscate(self, text: str, config: Dict = {"threshold": 1.0}) -> str:
        if "threshold" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'threshold' parameter."
            )

        threshold = config["threshold"]

        result = self.analyser.calculator.calculate_surprisal(text)
        surprisals = result["surprisals"]

        for i, s in enumerate(surprisals):
            if s > threshold:
                result["tokens"][i] = self.obfuscation_replacement
        print(result)

        tokens_as_ids = self.analyser.calculator.tokenizer.convert_tokens_to_ids(
            result["tokens"]
        )
        decoded = self.analyser.calculator.tokenizer.decode(tokens_as_ids)

        return decoded

    def analyse(self, texts: Union[List[str], str], save_plot_to: str = "dist.png"):
        self.analyser.get_distribution(texts, save_to=save_plot_to)
        mean_surp = self.analyser.get_mean(texts)
        median_surp = self.analyser.get_median(texts)

        print("====")
        print("Surprisal Analysis")
        print(f"Mean: {mean_surp}")
        print(f"Median: {median_surp}")
        print(f"Distribution saved to: {save_plot_to}")
        print("====")


if __name__ == "__main__":
    # just some code for testing. ignore!
    obf = SurprisalObfuscator()
    sample = """
          Discover the cosmos! Each day a different image or photograph 
          of our fascinating universe is featured, along with a brief explanation 
          written by a professional astronomer.
    """
    obf.analyse(sample, save_plot_to="dist.png")
    result = obf.obfuscate(sample, threshold=4)
    print(result)
