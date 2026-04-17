import random
from pathlib import Path
from datasets import load_dataset
from tmallet import TMallet  
from tmallet.obfuscators.shannon.impl import ShannonAnalyser, ShannonVisualiser

HF_DATASET  = "codelion/fineweb-edu-1B"
HF_CONFIG   = None
HF_SPLIT    = "train"
NUM_SAMPLES = 100        # number of sentences to analyse
MIN_WORDS   = 6         # skip very short / empty lines
RANDOM_SEED = 42

OUTPUT_DIR        = Path("./examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIST_PNG_PATH     = OUTPUT_DIR / "mutual_info_distribution.png"
HTML_OUTPUT_PATH  = OUTPUT_DIR / "distribution_heatmap.html"


def sample_texts_from_hf(
    dataset: str,
    config: str,
    split: str,
    n: int,
    min_words: int = 6,
    seed: int | None = 42,
) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0

    ds = load_dataset(dataset, config, split=split, streaming=True)

    for item in ds:
        text: str = item.get("text", "").strip()
        if not text or len(text.split()) < min_words:
            continue

        count += 1
        if len(reservoir) < n:
            reservoir.append(text)
        else:
            # reservoir sampling (Algorithm R)
            idx = rng.randint(0, count - 1)
            if idx < n:
                reservoir[idx] = text

    if len(reservoir) < n:
        print(
            f"Warning: only found {len(reservoir)} qualifying sentences "
            f"(requested {n})."
        )

    rng.shuffle(reservoir)
    return reservoir

if __name__ == "__main__":
    print(f"Sampling {NUM_SAMPLES} sentences from '{HF_DATASET}/{HF_CONFIG}' …")
    sample_texts = sample_texts_from_hf(
        HF_DATASET, HF_CONFIG, HF_SPLIT, NUM_SAMPLES, MIN_WORDS, RANDOM_SEED
    )
    print(f"{len(sample_texts)} sentences collected.\n")

    analyser   = ShannonAnalyser()
    visualiser = ShannonVisualiser()

    # compute MI distribution over all sampled sentences
    processed_texts = analyser.get_distribution_by_word(sample_texts, DIST_PNG_PATH)

    for text in processed_texts:
        print("====")
        print(text)
        print("====")

    # build per-sentence word / MI lists for the heatmap
    words = [[w.word for w in text.word_stats] for text in processed_texts]
    mi    = [[w.mutual_information for w in text.word_stats] for text in processed_texts]

    heatmap = visualiser.display_sentence_heatmap(words, mi)
    print(heatmap)

    HTML_OUTPUT_PATH.write_text(f"<html><body>{heatmap}</body></html>", encoding="utf-8")
    print(f"\nHeatmap written to: {HTML_OUTPUT_PATH}")
    print(f"Distribution plot : {DIST_PNG_PATH}")

