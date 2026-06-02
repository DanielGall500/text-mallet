from tmallet import TMallet

# -- Obfuscation Config --
algorithm = "scramble-BoW"
config = {
    "level": "document",
}

sample = "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe."

# -- Load Text Mallet and Obfuscate --
tmallet = TMallet(lang="en", prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)

obfuscated_text_sample = tmallet.obfuscate(sample)
print(obfuscated_text_sample)
