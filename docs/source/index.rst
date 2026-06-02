Text Mallet Documentation
=========================

**text-mallet** is a toolkit for transforming text into obfuscated or
derived formats while preserving utility for downstream NLP tasks such
as classification, retrieval, and topic modeling.

The package focuses on reducing the risk of privacy or copyright
infringement by degrading reconstructable information, while retaining task-relevant signals.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   usage/basics
   usage/configurations
   usage/datasets

.. toctree::
   :maxdepth: 1
   :caption: Obfuscation Methods

   methods/structural_hier
   methods/bag_of_words
   methods/pos_filtering
   methods/mutual_information

.. toctree::
   :maxdepth: 2
   :caption: API

   pipeline

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   about


Overview
--------

Text can be transformed along multiple linguistic dimensions:

- **Word Forms** (surface character sequences)
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
