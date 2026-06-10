from pathlib import Path

import torch

from tmallet import TMallet

torch.set_num_threads(1)

# -- Parameters --
OUTPUT_FOLDER = "test-obfuscation-results"
OBFUS_TYPE = "mutual-info-obfus"

# TESTING
CHUNK_SIZE = 16
BATCH_SIZE = 8
START_INDEX = 0
NUM_SAMPLES = 32
dataset_repo = "DanielGallagherIRE/fineweb-edu-10b"

# -- Obfuscation Config --
algorithm = "shannon"
config = {
    "threshold": [5, 7.5, 10, 12.5, 15],
    "bound": "lower",  # remove low MI words first, preserve signal
    "replacement_mechanism": "default",
    "max_context_length": 128,
    "output_mi_values": True,
}


def main():
    # -- Load Text Mallet and Obfuscate --
    lang = "en"
    prefer_gpu = True
    tmallet = TMallet(lang, prefer_gpu)

    print("====")
    print(f"Obfuscating {algorithm}")
    tmallet.load_obfuscator(algorithm, config)

    save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus" / algorithm
    tmallet.obfuscate_dataset_by_chunk(
        dataset_repo=dataset_repo,
        column="text",
        column_obfuscated=f"text_{algorithm}",
        save_chunks_to_folder=save_chunks_to,
        dataset_config=None,
        dataset_split="train",
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE,
        start_index=START_INDEX,
        num_samples=NUM_SAMPLES,
    )
    print(f"Finished obfuscation of {algorithm}")
    print("====")


print("Done!")


if __name__ == "__main__":
    main()
