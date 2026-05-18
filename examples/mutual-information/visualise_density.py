import random
from pathlib import Path
from datasets import load_dataset
from tmallet import TMallet  
from tmallet.obfuscators import ShannonBERT

HF_DATASET  = "codelion/fineweb-edu-1B"
HF_CONFIG   = None
HF_SPLIT    = "train"
NUM_SAMPLES = 2        # number of texts to analyse
MIN_WORDS   = 6         # skip very short / empty lines
RANDOM_SEED = 42

OUTPUT_DIR        = Path("./")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIST_PNG_PATH_EN  = OUTPUT_DIR / "mutual_info_distribution_en.png"
DIST_PNG_PATH_DE  = OUTPUT_DIR / "mutual_info_distribution_de.png"
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
    print(f"Gathering {NUM_SAMPLES} sample(s) from '{HF_DATASET}/{HF_CONFIG}' …")
    """
    sample_texts = sample_texts_from_hf(
        HF_DATASET, HF_CONFIG, HF_SPLIT, NUM_SAMPLES, MIN_WORDS, RANDOM_SEED
    )
    """

    sample_texts = load_dataset(HF_DATASET, split=HF_SPLIT).select(range(10))["text"]
    print(sample_texts)

    print(f"{len(sample_texts)} sentences collected.\n")
    print("Samples: ", len(sample_texts))
    print(sample_texts)

    analyser_en = ShannonBERT(lang="en", prefer_gpu=False)

    # compute MI distribution over all sampled sentences
    # also plots distribution to DIST_PNG_PATH
    mi = analyser_en.get_distribution_by_word(
            sample_texts, DIST_PNG_PATH_EN
    )

    """
    analyser_de = ShannonAnalyser(lang="de", prefer_gpu=False)
    mi = analyser_de.get_distribution_by_word(
            sample_texts_de, DIST_PNG_PATH_DE
    )
    """
