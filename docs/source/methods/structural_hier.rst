=====================
Hierarchical Scramble
=====================

The **Hierarchical Scramble** algorithm (``"scramble-hier"``) obfuscates text by shuffling its grammatical structure rather than masking or substituting words. By parsing sentences into dependency trees using spaCy, it rearranges tokens at various hierarchical levels (left and right branching child nodes). This method is ideal for tasks where you need to preserve the exact vocabulary and word frequencies of the original text while completely disrupting syntactic patterns, grammatical structures, and authorship stylometry. It is a lighter form of structural obfuscation when compared to linear scrambling (i.e. bag of words).

Configuration
-------------

To use the Hierarchical Scramble filter, initialize the obfuscator with the ``"scramble-hier"`` algorithm string and provide a configuration dictionary.

* **strength** (str or list of str): Determines the severity of the structural rearrangement.

  * ``"weak"``: Traverses the dependency tree and independently shuffles left-branching and right-branching sibling nodes. The overall left-to-right orientation of phrases relative to their parent heads is preserved.
  * ``"strong"``: Shuffles sibling nodes, introduces a 50% probability to randomly flip node directions (Left $\leftrightarrow$ Right), and reverses child-node placement order during sentence linearization.

  .. note::
     You can pass a single string (e.g., ``"weak"``) to get a flat string output, or a list of strengths (e.g., ``["weak", "strong"]``) to receive a nested dictionary containing results for all configured variations under the ``"scramble-hier"`` key.

Example Usage
-------------

The following example demonstrates how to execute both a weak and a strong hierarchical scramble pass sequentially on a sample text.

.. code-block:: python

    from tmallet import TMallet

    # 1. Define configurations for different scrambling strengths
    algorithm = "scramble-hier"
    config_weak = {
        "strength": "weak",
    }
    config_strong = {
        "strength": "strong",
    }

    # 2. Define Sample Text
    sample = "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe."

    # 3. Load Text Mallet and Obfuscate through both configurations
    for config in [config_weak, config_strong]:
        tmallet = TMallet(lang="en", prefer_gpu=True)
        tmallet.load_obfuscator(algorithm, config)

        obfuscated_text_sample = tmallet.obfuscate(sample)

        print(f"==Results ({config['strength']})==")
        print(obfuscated_text_sample)

Expected Output
---------------

Notice how the "weak" scramble alters the local placement of descriptors but retains a vaguely recognizable progression, while the "strong" scramble completely shatters phrase order and flips clauses.

.. code-block:: text

    ==Results (weak)==
    Leipzig is. most populous the city in the German state of Saxony The city
    has a population of 633,592 residents as of 31 December 2025. It is part
    of Central German the Metropolitan Region and. the eighth - largest city
    in Germany Leipzig is located about 150 km) southwest of Berlin (, in southernmost the part)
    (of the North German Plain Leipzig the Bay 90 mi. at the confluence of the White Elster its
    tributaries Pleiße and Parthe and

    ==Results (strong)==
    German of Saxony state the in Leipzig. most populous city the is of 2025 December 31 as.
    The city has population residents 633,592 of a It. and of the Central German Metropolitan Region part is the
    largest - eighth Germany in city located the southernmost North German Plain the of part () Bay Leipzig
    the in, km () 90 mi about 150 of Berlin southwest the confluence of its tributaries Pleiße Parthe and
    White and Elster the at is. Leipzig
