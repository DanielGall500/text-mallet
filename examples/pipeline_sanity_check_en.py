from tmallet import TMallet
from tmallet.obfuscators.shannon.impl import ShannonAnalyser, ShannonVisualiser

obfuscation_techniques = {
    "lemmatize": {
        "algorithm": "lemmatize",
    },
    "Retain Nouns w/ POS": {
        "algorithm": "noun-retain",
        "replacement_mechanism": "DELETE",
    },
    "Retain Nouns & PropNs w/ POS": {
        "algorithm": "noun-propn-retain",
        "replacement_mechanism": "DELETE",
    },
    "Remove Nouns w/ POS": {
        "algorithm": "noun-remove",
        "replacement_mechanism": "DELETE",
    },
    "Remove Nouns & PropNs w/ POS": {
        "algorithm": "noun-propn-remove",
        "replacement_mechanism": "DELETE",
    },
    "Retain Nouns w/ DEFAULT": {
        "algorithm": "noun-retain",
        "replacement_mechanism": "DEFAULT",
    },
    "Retain Nouns PropNs w/ DEFAULT": {
        "algorithm": "noun-propn-retain",
        "replacement_mechanism": "DEFAULT",
    },
    "Remove Nouns w/ DEFAULT": {
        "algorithm": "noun-remove",
        "replacement_mechanism": "DEFAULT",
    },
    "Remove Nouns PropNs w/ DEFAULT": {
        "algorithm": "noun-propn-remove",
        "replacement_mechanism": "DEFAULT",
    },
    "Retain Nouns w/ POS": {
        "algorithm": "noun-retain",
        "replacement_mechanism": "POS",
    },
    "Retain Nouns PropNs w/ POS": {
        "algorithm": "noun-propn-retain",
        "replacement_mechanism": "POS",
    },
    "Remove Nouns w/ POS": {
        "algorithm": "noun-remove",
        "replacement_mechanism": "POS",
    },
    "Retain Nouns PropNs w/ POS": {
        "algorithm": "noun-propn-remove",
        "replacement_mechanism": "POS",
    },
    "scramble-hier-weak": {
        "algorithm": "scramble-hier-weak",
    },
    "scramble-hier-strong": {
        "algorithm": "scramble-hier-strong",
    },
    "scramble-BoW-sentence": {
        "algorithm": "scramble-BoW-sentence",
    },
    "scramble-BoW-document": {
        "algorithm": "scramble-BoW-document",
    },
    "Shannon (10, DELETE)": {
        "algorithm": "shannon",
        "threshold": 8,
        "replacement_mechanism": "DELETE",
        "as_upper_bound": True,
        "as_lower_bound": True,
        "output_mi_values": True,
    },
    "Shannon (10, DEFAULT)": {
        "algorithm": "shannon",
        "threshold": 8,
        "replacement_mechanism": "DEFAULT",
        "as_upper_bound": True,
        "as_lower_bound": True,
        "output_mi_values": True,
    },
    "Shannon (10, POS)": {
        "algorithm": "shannon",
        "threshold": 8,
        "replacement_mechanism": "POS",
        "as_upper_bound": True,
        "as_lower_bound": True,
        "output_mi_values": True,
    },
}


def main():
    sample_texts = [
        "Three-dimensional printing is being used to make metal parts for aircraft and space vehicles."
    ]
    language = "en"
    prefer_gpu = True

    tmallet = TMallet(language, prefer_gpu)

    for technique, config in obfuscation_techniques.items():
        if "Shannon" not in technique:
            pass

        tmallet.load_obfuscator(config)

        for text in sample_texts:
            obfuscated_text = tmallet.obfuscate(text)

            if "Shannon" in technique:
                for threshold, results in obfuscated_text.items():
                    print("==Shannon==")
                    print(threshold)
                    print("Threshold: ", threshold)
                    if "as_lower_bound" in results.keys():
                        print("Lower Bounded: ", results["as_lower_bound"])
                    if "as_upper_bound" in results.keys():
                        print("Upper Bounded: ", results["as_upper_bound"])
                        print("===========")
                    if "mi_values" in results.keys():
                        print("MI Values: ", [round(x,2) for x in results["mi_values"]])
                        print("===========")
            else:
                print(f"===={technique}====")
                print(obfuscated_text)
                print("================")
                print("\n\n")


if __name__ == "__main__":
    main()
