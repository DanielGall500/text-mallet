from pathlib import Path

from tmallet import TMallet

# -- Parameters --
OUTPUT_FOLDER = "results"
OBFUS_TYPE = "structural-final-test"

CHUNK_SIZE = 20
START_INDEX = 200
NUM_SAMPLES = 60
BATCH_SIZE = 10
save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus"
dataset_repo = "codelion/fineweb-edu-1B"

# -- Obfuscation Config --
algorithm = "scramble-BoW"
config = {
    "level": ["sentence", "document"],
}

# -- Load Text Mallet and Obfuscate --
tmallet = TMallet(lang="en", prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)
obfuscated_text_by_chunk = tmallet.obfuscate_dataset_by_chunk(
    dataset_repo=dataset_repo,
    column="text",
    column_obfuscated="text_BoW",
    save_chunks_to_folder=save_chunks_to,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
    start_index=START_INDEX,
    num_samples=NUM_SAMPLES,
)
print(obfuscated_text_by_chunk.to_pandas().describe())
