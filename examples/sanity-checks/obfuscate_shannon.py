import json
from pathlib import Path

from datasets import load_dataset

from tmallet import TMallet

# -- Parameters --
test_one_sample = False

OUTPUT_FOLDER = "results"
OBFUS_TYPE = "mutual-info"

CHUNK_SIZE = 5
BATCH_SIZE = 1
save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus"

subset_n = 5
dataset = "codelion/fineweb-edu-1B"

# -- Obfuscation Config --
algorithm = "shannon"
config = {
    "threshold": 8,
    "bound": "upper",
    "replacement_mechanism": ["delete", "default", "POS"],
    "max_context_length": 128,
    "output_mi_values": True,
}

# -- Load Text Mallet and Obfuscate --
tmallet = TMallet(lang="en", prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)
obfuscated_text_by_chunk = tmallet.obfuscate_dataset_by_chunk(
    dataset_repo=dataset,
    column="text",
    column_obfuscated="text_shannon",
    save_chunks_to_folder=save_chunks_to,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
)
print(obfuscated_text_by_chunk.to_pandas().describe())
