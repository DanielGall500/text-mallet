=============
POS Filtering
=============

The **POS Filtering** algorithm allows you to selectively retain or remove specific Part-of-Speech (POS) tags in your text while obfuscating the remainder.

Configuration
-------------

To use the POS Filter, initialize the obfuscator with the ``"pos-filter"`` algorithm string and provide a configuration dictionary.

* **filter_type** (list of str): Determines the filtering behavior. For example, ``"retain"`` keeps the specified tags and obfuscates the rest, while ``"remove"`` removes the specified tags.
* **pos_tags** (list of str): A list of Universal Part-of-Speech (UPOS) tags to target (e.g. ``["NOUN", "PROPN"]``).
* **replacement_mechanism** (list of str): Defines how the obfuscated words are masked. Setting this to ``["POS"]`` replaces the hidden words with their respective UPOS tags. Can also be ``"default"`` for a default token (underscore) or ``"delete"`` to just delete the word entirely.
* **seed** (int): An integer seed to ensure reproducibility across runs.

Example Usage
-------------

The following example demonstrates how to configure the POS filter to retain nouns and proper nouns in both English and German text. All other words are replaced with their underlying POS tags.

.. code-block:: python

    from tmallet import TMallet

    # 1. Define the Obfuscation Configuration
    algorithm = "pos-filter"
    config = {
        "filter_type": ["retain"],
        "pos_tags": ["NOUN", "PROPN"],
        "replacement_mechanism": ["POS"],
        "seed": 100,
    }

    # 2. Define Sample Texts
    test_languages = {
        "en": "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe.",
        "de": "Leipzig ist eine kreisfreie Stadt sowie mit 611.850 Einwohnern (31. Dezember 2024, laut Statistischem Landesamt des Freistaates Sachsen) bzw. 633 592 Einwohnern (laut Melderegister am 31. Dezember 2025) die einwohnerreichste Stadt im Freistaat Sachsen. Sie belegte 2024 in der Liste der Großstädte in Deutschland den achten Rang. Für Mitteldeutschland ist sie ein historisches Zentrum der Wirtschaft, des Handels und Verkehrs, der Verwaltung, Kultur und Bildung sowie gegenwärtig ein Zentrum für die „Kreativszene“ und eine wichtige Messe- und Universitätsstadt.",
    }

    # 3. Load Text Mallet and Obfuscate
    for lang, sample in test_languages.items():
        tmallet = TMallet(lang=lang, prefer_gpu=False)
        tmallet.load_obfuscator(algorithm, config)
        obfuscated_text_sample = tmallet.obfuscate(sample)

        print(f"==Result ({lang})==")
        print(obfuscated_text_sample)

Expected Output
---------------

.. code-block:: text

    ==Result (en)==
    Leipzig AUX DET ADV ADJ city ADP DET ADJ state ADP Saxony PUNCT DET city AUX DET population ADP NUM residents ADP ADP NUM December NUM PUNCT PRON AUX DET ADV PUNCT ADJ city ADP Germany CCONJ AUX part ADP DET ADJ ADJ Metropolitan Region PUNCT Leipzig AUX VERB ADP NUM km PUNCT NUM mi PUNCT ADV ADP Berlin PUNCT ADP DET ADJ part ADP DET ADJ German Plain PUNCT DET Leipzig Bay PUNCT PUNCT ADP DET confluence ADP DET White Elster CCONJ PRON tributaries Pleiße CCONJ Parthe PUNCT

    ==Result (de)==
    Leipzig AUX DET ADJ Stadt CCONJ ADP NUM Einwohnern PUNCT ADJ Dezember NUM PUNCT ADP ADJ Landesamt DET Freistaates Sachsen PUNCT CCONJ NUM SPACE NUM Einwohnern PUNCT ADP Melderegister ADP ADJ Dezember NUM PUNCT DET ADJ Stadt ADP Freistaat Sachsen PUNCT PRON VERB NUM ADP DET Liste DET Großstädte ADP Deutschland DET ADJ Rang PUNCT ADP Mitteldeutschland AUX PRON DET ADJ Zentrum DET Wirtschaft PUNCT DET Handels CCONJ Verkehrs PUNCT DET Verwaltung PUNCT Kultur CCONJ Bildung CCONJ ADV DET Zentrum ADP DET PUNCT Kreativszene PUNCT CCONJ DET ADJ X CCONJ Universitätsstadt PUNCT
