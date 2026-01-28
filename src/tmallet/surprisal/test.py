from tmallet.surprisal.analysis import SurprisalAnalyser
from tmallet.surprisal.visualise import SurprisalVisualiser
from IPython.display import HTML
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset(
        "DanielGallagherIRE/fineweb-edu-1B-obfuscated", split="train"
    )
    print(dataset)
    samples = dataset["text"][:2]

    example = """
    Discover the cosmos! Each day a different image or photograph of our 
    fascinating universe is featured, along with a brief explanation written 
    by a professional astronomer. 2012 June 23. Explanation: As seen from 
    Frösön island in northern Sweden the Sun did set a day after the summer
    solstice. From that location below the arctic circle it settled slowly 
    behind the northern horizon. During the sunset's final minute, this 
    remarkable sequence of 7 images follows the distorted edge of the solar 
    disk as it just disappears against a distant tree line, capturing both a 
    green and blue flash. Not a myth even in a land of runes, the colorful 
    but elusive glints are caused by atmospheric refraction enhanced by long, 
    low, sight lines and strong atmospheric temperature gradients.
    """
    analyser = SurprisalAnalyser()
    plotter = SurprisalVisualiser()

    result_WL = analyser.get_distribution_surprisal_by_word(example)
    result_TL = analyser.get_distribution_surprisal_by_token(example)

    print(result_WL)
    print("====")
    print(result_TL)

    html_obj_WL = plotter.display_sentence_heatmap(
        result_WL["words"], result_WL["surprisals"]
    )
    html_obj_TL = plotter.display_sentence_heatmap(
        result_TL["tokens"], result_TL["surprisals"]
    )
    html_obj = HTML(html_obj_WL.data + "<br><br><br>" + html_obj_TL.data)
    # Save HTML string to a file
    with open("surprisal_heatmap.html", "w") as f:
        f.write(html_obj.data)  # .data contains the raw HTML string

    # Then open it in your default browser
    import webbrowser

    webbrowser.open("surprisal_heatmap.html")

    """
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
    """
