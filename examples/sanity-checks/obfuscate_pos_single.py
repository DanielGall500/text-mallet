from tmallet import TMallet

# -- Obfuscation Config --
algorithm = "pos-filter"
config = {
    "filter_type": ["retain"],
    "pos_tags": ["NOUN", "PROPN"],
    "replacement_mechanism": ["POS"],
    "seed": 100,
}

test_languages = {
    "en": "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe.",
    "de": "Leipzig ist eine kreisfreie Stadt sowie mit 611.850 Einwohnern (31. Dezember 2024, laut Statistischem Landesamt des Freistaates Sachsen) bzw. 633 592 Einwohnern (laut Melderegister am 31. Dezember 2025) die einwohnerreichste Stadt im Freistaat Sachsen. Sie belegte 2024 in der Liste der Großstädte in Deutschland den achten Rang. Für Mitteldeutschland ist sie ein historisches Zentrum der Wirtschaft, des Handels und Verkehrs, der Verwaltung, Kultur und Bildung sowie gegenwärtig ein Zentrum für die „Kreativszene“ und eine wichtige Messe- und Universitätsstadt.",
}

# -- Load Text Mallet and Obfuscate --
for lang, sample in test_languages.items():
    tmallet = TMallet(lang=lang, prefer_gpu=True)
    tmallet.load_obfuscator(algorithm, config)

    obfuscated_text_sample = tmallet.obfuscate(sample)
    print("==Result==")
    print(obfuscated_text_sample)
