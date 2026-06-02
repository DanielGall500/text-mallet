==============
Linear Scramble
==============

The **Linear Scramble** algorithm (``"scramble-BoW"``) obfuscates text using a flat Bag-of-Words (BoW) randomisation approach. Unlike structural scramblers, this method completely ignores syntax and grammar, randomly shuffling the linear order of words.

Since it relies purely on basic string splitting and Python's native pseudorandom number generation, **it does not require heavy external NLP frameworks like spaCy or BERT**. This makes it an incredibly fast, lightweight option for baseline anonymization or vocabulary-preserving obfuscation.

Configuration
-------------

To use the Linear Scramble filter, initialise the obfuscator with the ``"scramble-BoW"`` algorithm string and provide a configuration dictionary.

* **level** (str or list of str): Dictates the structural boundary within which words are shuffled.

  * ``"sentence"``: Tokenizes the text into sentences first, then shuffles the words *within* each individual sentence. Sentence boundaries are preserved.
  * ``"document"``: Shuffles all words across the entire text input simultaneously, destroying sentence boundaries completely.

  .. note::
     You can pass a single string (e.g., ``"document"``) to return a flat string, or a list of levels (e.g., ``["sentence", "document"]``) to receive a nested dictionary containing results for all configured variations under the ``"scramble-linear"`` key.

* **seed** (int): An integer seed passed to the random number generator to ensure reproducible shuffles across runs.

Example Usage
-------------

The following example demonstrates how to execute a document-level linear scramble on a sample text.

.. code-block:: python

    from tmallet import TMallet

    # 1. Define the Obfuscation Configuration
    algorithm = "scramble-BoW"
    config = {
        "level": "document",
    }

    # 2. Define Sample Text
    sample = "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe."

    # 3. Load Text Mallet and Obfuscate
    tmallet = TMallet(lang="en", prefer_gpu=True)
    tmallet.load_obfuscator(algorithm, config)

    obfuscated_text_sample = tmallet.obfuscate(sample)
    print(obfuscated_text_sample)

Expected Output
---------------

Notice that punctuation characters remain attached to the words they modified during basic token splitting, and the entire paragraph structure is flattened into a single randomized sequence.

.. code-block:: text

    southernmost 150 eighth-largest most the (90 Central located Bay), Parthe. mi) in city of German Germany part population the is Plain of populous and as Leipzig the German the the km White its Metropolitan Berlin, and Leipzig in the confluence 2025. of of city It Elster is Region. December tributaries The state Saxony. (the southwest the residents the of city Leipzig German is in is Pleiße of a has at part about of 31 and North 633,592
