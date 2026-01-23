from tmallet.surprisal.surprisal import BERTSurprisalCalculator
from tmallet.surprisal.plot import SurprisalPlotter
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset(
        "DanielGallagherIRE/fineweb-edu-1B-obfuscated", split="train"
    )
    print(dataset)
    samples = dataset["text"][:2]

    calculator = BERTSurprisalCalculator()

    plotter = SurprisalPlotter()

    # text = "The quick brown fox jumps over the lazy dog."
    surprisals = []
    results = []
    for s in samples:
        result = calculator.calculate_surprisal(s)
        surprisals.append(result["surprisals"])
        results.append(result)

    s_threshold = 5

    to_obfuscate = [
        [s > s_threshold for s in result["surprisals"]] for result in results
    ]

    for result, to_obfuscate_sample in zip(results, to_obfuscate):
        for i, is_obfuscated in enumerate(to_obfuscate_sample):
            if is_obfuscated:
                result["tokens"][i] = "[OBF]"

    plotter.prepare_data(
        surprisals=surprisals,
    )

    hist_plot = plotter.plot_density()
    hist_plot.save("hist.png")
