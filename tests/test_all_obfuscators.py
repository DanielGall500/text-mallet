from tmallet import TMallet

sample_text = """
Leipzig has been a trade city since at least the time of the Holy Roman Empire. 
"""
mallet = TMallet()
obfuscation_techniques = [
    "noun",
    "noun-propn",
    "no-noun",
    "no-noun-propn",
    "lemmatization",
    "scramble-BoW",
    "scramble-BoW-by-sentence",
    "scramble-shuffle-siblings",
    "scramble-reverse-head",
    # "shannon",
]


for technique in obfuscation_techniques:
    print(technique)
    obfuscated_text = mallet.obfuscate(sample_text, config={"algorithm": technique})
    print("====")
    print(technique, obfuscated_text)
    print("====")


"""
obfuscations = {
    "Lemmas Only": output_lemma,
    "Linear (No Split)": output_linear,
    "Linear (Split by Sentence)": output_linear_within_sent,
    "Hierarchical (Shuffle Siblings Randomly)": output_shuffle_siblings,
    "Hierarchical (Reverse Head Direction Randomly)": output_rev_head_direction,
    "Nouns & Proper Nouns Only": output_nouns_and_propn_only,
    "Nouns Only": output_nouns_only,
    "No Nouns": output_no_nouns,
    "No Nouns Nor Proper Nouns": output_no_nouns_or_propn,
    "Nouns & Proper Nouns Only (incl. POS)": output_nouns_and_propn_only_replace,
    "Nouns Only (incl. POS)": output_nouns_only_replace,
    "No Nouns (incl. POS)": output_no_nouns_replace,
    "No Nouns Nor Proper Nouns (incl. POS)": output_no_nouns_or_propn_replace
    "Shannon": output_no_nouns_or_propn_replace
}


# Display results
print("=" * 80)
print("ORIGINAL TEXT")
print("=" * 80)
print(text)
print("\n")

for method_name, obfuscated_text in obfuscations.items():
    print("=" * 80)
    print(f"OBFUSCATION METHOD: {method_name}")
    print("=" * 80)
    print(obfuscated_text)
    print("\n")
"""
