Text Mallet Documentation
=========================

**text-mallet** is a toolkit for transforming text into obfuscated or
derived formats while preserving utility for downstream NLP tasks such
as classification, retrieval, and topic modeling.

The package focuses on reducing the risk of privacy or copyright
infringement by degrading reconstructable linguistic information,
while retaining task-relevant signals.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage/basics
   usage/datasets
   usage/large_datasets

.. toctree::
   :maxdepth: 1
   :caption: Obfuscation Methods

   methods/lemmatization
   methods/structural
   methods/pos_filtering
   methods/mutual_information

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   about
   contributing
   changelog
   license


Overview
--------

Text can be transformed along multiple linguistic dimensions:

- **Word Forms** (surface character sequences)
- **Root Forms** (lemmas)
- **Syntactic and Morphological Features**
- **Semantic Content**
- **Grammatical Relations**
- **Sequence Structure**

Each of these contributes information to the final text. ``text-mallet``
provides mechanisms to selectively erode this information, producing
representations that are less human-readable but still useful for
machine learning tasks.

Different languages rely on these dimensions differently. For example,
English depends heavily on word order, while German relies more on
morphological variation.


Why Obfuscate Text?
-------------------

Many NLP tasks do not require fully reconstructable text. Tasks such as:

- Text classification
- Semantic similarity
- Topic modeling
- Information retrieval

can often operate effectively on degraded or transformed inputs.

This package enables:

- Possible use of sensitive or copyrighted data without exposing raw text
- Reduced risk of reconstruction from adversarial attacks (e.g. embedding inversion) or thorugh model outputs

Rather than replacing clean data, obfuscated text is intended to
*complement* existing datasets.


Quick Example
-------------

The following example demonstrates how to use ``TMallet`` to obfuscate text
based on part-of-speech replacement.

.. code-block:: python

    from tmallet import TMallet

    mallet = TMallet()

    text = """
    A Soyuz rocket launched two Galileo satellites into orbit on Friday,
    marking a crucial step for Europe’s planned navigation system,
    operator Arianespace announced.
    """

    config = {
        "algorithm": "retain-noun-propn",
        "replacement_mechanism": "POS"
    }

    obfuscated_text = mallet.obfuscate(text, config)


Output
------

.. code-block:: text

    DET Soyuz rocket VERB NUM Galileo satellites ADP orbit ADP Friday PUNCT
    VERB DET ADJ step ADP Europe PART VERB navigation system PUNCT
    operator Arianespace VERB

Core Obfuscation Strategies
--------------------------

``text-mallet`` provides four primary approaches:

1. **Lemmatization**
   - Light obfuscation
   - Removes stylistic variation

2. **Scrambling**
   - Alters word order
   - Ranges from structure-aware to bag-of-words randomization

3. **Part-of-Speech Filtering**
   - Removes or isolates word classes
   - Medium to strong obfuscation

4. **Mutual Information Filtering**
   - Uses Shannon entropy approximation
   - Filters words based on contextual importance
   - Highly configurable strength


Design Goals
------------

- Preserve task-relevant signal
- Minimize reconstructability of original text
- Provide configurable and composable text transformations


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
