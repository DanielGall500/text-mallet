from tmallet import TMallet

# -- Obfuscation Config --
algorithm = "shannon"
config = {
    "threshold": 9.5,
    "bound": "upper",
    "replacement_mechanism": "default",
    "max_context_length": 128,
}
sample = "Data obfuscation is the process of modifying sensitive data in such a way that it is of no or little value to unauthorized intruders while still being usable by software or authorized personnel. Data masking can also be referred as anonymization, or tokenization, depending on different context."

tmallet = TMallet(lang="en", prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)

obfuscated_text_sample = tmallet.obfuscate(sample)
print("==Result==")
print(obfuscated_text_sample)
