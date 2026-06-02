from tmallet import TMallet
from tmallet.obfuscators.shannon.visualise import ShannonVisualiser
from datasets import load_dataset
from pathlib import Path
import json

# -- Parameters --
test_one_sample = True

OUTPUT_FOLDER = "results"
OBFUS_TYPE = "mutual-info"

CHUNK_SIZE = 5
BATCH_SIZE = 1
save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus"

subset_n = 10
dataset_repo = "codelion/fineweb-edu-1B"
dataset = load_dataset(dataset_repo)["train"]
dataset = dataset.select(range(subset_n))
print(dataset.to_pandas().head())

# -- Obfuscation Config --
algorithm = "shannon"
config_higher_max_context_length = {
    "threshold": [7.5],
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": ["DELETE", "DEFAULT", "POS"],
    "max_context_length": 8192
}
config_lower_max_context_length = {
    "threshold": [7.5],
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": ["DELETE", "DEFAULT", "POS"],
    "max_context_length": 20
}

test_languages = {
    "en": "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe.", 
    "de": "Leipzig ist eine kreisfreie Stadt sowie mit 611.850 Einwohnern (31. Dezember 2024, laut Statistischem Landesamt des Freistaates Sachsen) bzw. 633 592 Einwohnern (laut Melderegister am 31. Dezember 2025) die einwohnerreichste Stadt im Freistaat Sachsen. Sie belegte 2024 in der Liste der Großstädte in Deutschland den achten Rang. Für Mitteldeutschland ist sie ein historisches Zentrum der Wirtschaft, des Handels und Verkehrs, der Verwaltung, Kultur und Bildung sowie gegenwärtig ein Zentrum für die „Kreativszene“ und eine wichtige Messe- und Universitätsstadt."
}

# -- Load Text Mallet and Obfuscate --
all_html = []
for lang, sample in test_languages.items():
    tmallet = TMallet(lang=lang, prefer_gpu=True)
    tmallet.load_obfuscator(algorithm, config_higher_max_context_length)

    obfuscated_text_sample = tmallet.obfuscate(sample)
    print(json.dumps(obfuscated_text_sample, indent=4))
    vis_html = tmallet.get_active_obfuscator().visualise()
    all_html.append(vis_html)

    tmallet.load_obfuscator(algorithm, config_lower_max_context_length)

    obfuscated_text_sample = tmallet.obfuscate(sample)
    print(json.dumps(obfuscated_text_sample, indent=4))
    vis_html = tmallet.get_active_obfuscator().visualise()
    all_html.append(vis_html)
        
    with open(f"check_context_{lang}.html", "w") as html_file:
        html_file.write(" ".join(all_html))
