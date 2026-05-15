from tmallet import TMallet
from datasets import load_dataset
from pathlib import Path
import json

# -- Parameters --
test_one_sample = True

OUTPUT_FOLDER = "results"
OBFUS_TYPE = "structural"

CHUNK_SIZE = 5
BATCH_SIZE = 1
save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus"

subset_n = 10
dataset_repo = "codelion/fineweb-edu-1B"
dataset = load_dataset(dataset_repo)["train"]
dataset = dataset.select(range(subset_n))
print(dataset.to_pandas().head())

# -- Obfuscation Config --
algorithm = "scramble-hier"
config = {
    "strength": ["strong","weak"],
    "seed": 100,
}

test_languages = {
    "en": "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe.", 
    "de": "Leipzig ist eine kreisfreie Stadt sowie mit 611.850 Einwohnern (31. Dezember 2024, laut Statistischem Landesamt des Freistaates Sachsen) bzw. 633 592 Einwohnern (laut Melderegister am 31. Dezember 2025) die einwohnerreichste Stadt im Freistaat Sachsen. Sie belegte 2024 in der Liste der Großstädte in Deutschland den achten Rang. Für Mitteldeutschland ist sie ein historisches Zentrum der Wirtschaft, des Handels und Verkehrs, der Verwaltung, Kultur und Bildung sowie gegenwärtig ein Zentrum für die „Kreativszene“ und eine wichtige Messe- und Universitätsstadt."
}

# -- Load Text Mallet and Obfuscate --
for lang, sample in test_languages.items():
    tmallet = TMallet(lang=lang, prefer_gpu=True)
    tmallet.load_obfuscator(algorithm, config)

    if test_one_sample:
        obfuscated_text_sample = tmallet.obfuscate(sample)
        print(json.dumps(obfuscated_text_sample, indent=4))
    else:
        obfuscated_text_by_chunk = tmallet.obfuscate_dataset_by_chunk(
            dataset=dataset, 
            column="text", 
            column_obfuscated="text_shannon",
            config=config,
            save_chunks_to_folder=save_chunks_to,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
            num_proc=None,
            device="cuda"
        )
        print(obfuscated_text_by_chunk.to_pandas().describe())

