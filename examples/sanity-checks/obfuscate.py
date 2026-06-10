import json
from pathlib import Path

from tmallet import TMallet

# -- Parameters --
test_one_sample = False

OUTPUT_FOLDER = "results"
OBFUS_TYPE = "all"

CHUNK_SIZE = 5
BATCH_SIZE = 1
save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus"

subset_n = 10
dataset_repo = "codelion/fineweb-edu-1B"

# -- Obfuscation Config --
config_single = {
    "pos-filter": {
        "filter_type": "retain",
        "pos_tags": ["NOUN", "PROPN"],
        "replacement_mechanism": "POS",
    },
    "shannon": {
        "threshold": 10,
        "bound": "lower",
        "replacement_mechanism": "default",
    },
    "scramble-hier": {
        "algorithm": "scramble-hier",
        "strength": "weak",
    },
    "scramble-BoW": {
        "level": "document",
    },
}
config_multi = {
    "pos-filter": {
        "filter_type": "retain",
        "pos_tags": ["NOUN", "PROPN"],
        "replacement_mechanism": ["delete", "default", "POS"],
        "seed": 100,
    },
    "shannon": {
        "threshold": [10, 12, 15],
        "bound": ["lower", "upper"],
        "replacement_mechanism": ["default", "POS"],
    },
    "scramble-hier": {
        "algorithm": "scramble-hier",
        "strength": ["strong", "weak"],
        "seed": 100,
    },
    "scramble-BoW": {
        "level": ["sentence", "document"],
        "seed": 100,
    },
}
config_test_dataset = {
    "scramble-BoW": {
        "level": ["sentence", "document"],
        "seed": 100,
    },
}

sample = "Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context."


# -- Load Text Mallet and Obfuscate --
tmallet = TMallet(lang="en", prefer_gpu=True)

if test_one_sample:
    all_results = {}
    for config_type in [config_single]:
        for algorithm, config in config_type.items():
            tmallet.load_obfuscator(algorithm, config)

            obfuscated_text_sample = tmallet.obfuscate(sample)
            print("\n\n")
            print(f"====={algorithm}=====")
            if isinstance(obfuscated_text_sample, dict):
                print(json.dumps(obfuscated_text_sample, indent=4))
            else:
                print(obfuscated_text_sample)
                print("=====")
else:
    tmallet.load_obfuscator("scramble-BoW", config_test_dataset["scramble-BoW"])
    obfuscated_text_by_chunk = tmallet.obfuscate_dataset_by_chunk(
        dataset_repo=dataset_repo,
        column="text",
        column_obfuscated="text_obfus",
        save_chunks_to_folder=save_chunks_to,
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE,
    )
