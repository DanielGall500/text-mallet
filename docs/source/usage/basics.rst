======
Basics
======

Text can be transformed along multiple linguistic dimensions:

- **Word Forms** (surface character sequences)
- **Syntactic and Morphological Features**
- **Semantic Content**
- **Grammatical Relations**
- **Sequence Structure**

The perturbation of any given dimension naturally reduces the ability of any person or model to reconstruct said text.
``text-mallet`` thus provides mechanisms to selectively erode a text's information, producing
representations that are less human-readable but still useful for
machine learning tasks. Different languages rely on these dimensions differently. For example,
English depends heavily on word order, while German relies more on
morphological variation.


Why Obfuscate Text?
-------------------

Many encoder-based training tasks do not require fully reconstructable text in order to learn.
Tasks such as text classification, semantic similarity, topic modelling, and information retrieval can often operate effectively on degraded or transformed inputs.
This package thus enables the possible use of sensitive or copyrighted data without exposing raw text, as well as a reduced risk of reconstruction from adversarial attacks (e.g. embedding inversion) or through model outputs

Rather than replacing clean data, obfuscated text is intended to
*complement* existing datasets. Particularly for cases where there is a sufficient basis of publicly licensed data that may be accompanied by obfuscated, proprietary data in the pre-training mix.

Obfuscating A Text
------------------

Using ``text-mallet`` follows a simple three-step lifecycle:

1. **Initialise** a ``TMallet`` instance for your target language.
2. **Load** an obfuscation algorithm along with its configuration.
3. **Obfuscate** your text, or alternatively, a column in a Hugging Face dataset.

You first initialise a Text Mallet instance like so:

.. code-block:: python

   from tmallet import TMallet

   tmallet = TMallet(lang="en", prefer_gpu=True, model_type="lg")

We have three initial decisions to make, (1) the language ("en" or "de), (2) whether to use a GPU if available, and (3) the type of SpaCy model to use (defaults to large, i.e. "lg")
We then define the algorithm and configuration:

.. code-block:: python

   text = "Leipzig is the most populous city in the German state of Saxony."

   algorithm = "pos-filter"
   config = {
       "filter_type": "retain",
       "pos_tags": ["NOUN", "PROPN"],
       "replacement_mechanism": "default",
   }

   tmallet.load_obfuscator(algorithm, config)

Lastly, we can obfuscate the text simply like so:

.. code-block:: python

   obfuscated = tmallet.obfuscate(text)

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
