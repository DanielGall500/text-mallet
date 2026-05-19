Basics
======

``text-mallet`` is a lightweight text obfuscation package for transforming text into less human-readable forms while preserving partial utility for NLP tasks such as:

* Classification
* Retrieval
* Topic modelling
* Semantic similarity

The package currently supports:

* English
* German

Quick Start
-----------

Using ``text-mallet`` consists of three simple steps:

1. Create a ``TMallet`` instance
2. Load an obfuscation algorithm
3. Obfuscate text

Example
^^^^^^^^

.. code-block:: python

   from tmallet import TMallet

   text = "Leipzig is the most populous city in the German state of Saxony."

   algorithm = "pos-filter"

   config = {
       "filter_type": "retain",
       "pos_tags": ["NOUN", "PROPN"],
       "replacement_mechanism": "DEFAULT",
   }

   tmallet = TMallet(lang="en", prefer_gpu=True)

   tmallet.load_obfuscator(algorithm, config)

   obfuscated = tmallet.obfuscate(text)

   print(obfuscated)

Available Algorithms
--------------------

``text-mallet`` currently provides four general obfuscation approaches, which are adjusted based on your configuration.
For details on configuring any individual such method, please see their respective page in the docs.

Lemmatisation
^^^^^^^^^^^^^

Reduce words to their base forms.

.. code-block:: python

   algorithm = "lemmatize"

Scrambling
^^^^^^^^^^

Shuffle word order while preserving varying amounts of structure.

.. code-block:: python

   algorithm = "scramble-hier"

or

.. code-block:: python

   algorithm = "scramble-BoW"

Part-of-Speech Filtering
^^^^^^^^^^^^^^^^^^^^^^^^

Filter or retain selected POS tags.

.. code-block:: python

   algorithm = "pos-filter"

Mutual Information Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filter words based on an approximation of word importance.

.. code-block:: python

   algorithm = "shannon"

Dataset Obfuscation
-------------------

You can also obfuscate entire HuggingFace datasets.

.. code-block:: python

   obfuscated_dataset = tmallet.obfuscate_dataset(
       dataset=dataset,
       column="text",
       column_obfuscated="text_obfuscated",
       config=config,
       batch_size=100
   )
