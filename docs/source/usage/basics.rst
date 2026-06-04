===========
Quick Start
===========

Using ``text-mallet`` follows a simple three-step lifecycle:

1. **Initialise** a ``TMallet`` instance for your target language.
2. **Load** an obfuscation algorithm along with its configuration.
3. **Obfuscate** your text, or alternatively, a column in a Hugging Face dataset.

Example
^^^^^^^

.. code-block:: python

   from tmallet import TMallet

   text = "Leipzig is the most populous city in the German state of Saxony."

   # 1. Choose algorithm and define parameters
   algorithm = "pos-filter"
   config = {
       "filter_type": "retain",
       "pos_tags": ["NOUN", "PROPN"],
       "replacement_mechanism": "default",
   }

   # 2. Spin up the engine (enable GPU acceleration if available)
   tmallet = TMallet(lang="en", prefer_gpu=True)
   tmallet.load_obfuscator(algorithm, config)

   # 3. Transform the text
   obfuscated = tmallet.obfuscate(text)
   print(obfuscated)

Available Algorithms
--------------------

``text-mallet`` provides four core algorithmic approaches to text transformation. For exhaustive parameter constraints, see the :doc:`configurations` page.

Scrambling
^^^^^^^^^^

Shuffles word and token placements within specified boundaries. This can be done linearly or based on hierarchical dependency parsing trees.

.. code-block:: python

   # Linear Bag-of-Words shuffling
   algorithm = "scramble-BoW"

   # Syntactic dependency-tree structural shuffling
   algorithm = "scramble-hier"

Part-of-Speech Filtering
^^^^^^^^^^^^^^^^^^^^^^^^

Selectively strips out or retains words belonging to specific grammatical universal POS tag classes.

.. code-block:: python

   algorithm = "pos-filter"

Shannon Filtering
^^^^^^^^^^^^^^^^^

Computes approximations of pointwise mutual information for each word and its context, filtering out words falling outside target boundaries.

.. code-block:: python

   algorithm = "shannon"

Dataset Obfuscation
-------------------

For larger processing jobs, ``text-mallet`` scales directly with the Hugging Face ecosystem. Rather than processing text row-by-row, load your settings once and map them across an entire dataset dataset collection.

.. code-block:: python

   # 1. Set up the engine state
   tmallet.load_obfuscator(algorithm, config)

   # 2. Run batched execution mapping over the dataset
   obfuscated_dataset = tmallet.obfuscate_dataset(
       dataset=my_hf_dataset,
       column="text",
       column_obfuscated="text_obfuscated",
       batch_size=100,
       num_proc=4
   )
