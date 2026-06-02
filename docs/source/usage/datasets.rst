====================
Obfuscating Datasets
====================

When working with large-scale NLP pipelines, obfuscating text line-by-line in a simple loop can become a significant performance bottleneck. To solve this, TMallet integrates directly with the Hugging Face ecosystem, allowing you to run optimized obfuscation passes over entire datasets.

text-mallet offers two primary methods for dataset-level operations: a standard in-memory batch mapping approach and a fault-tolerant chunked streaming approach for massive datasets.

Standard In-Memory Mapping
---------------------------

The ``obfuscate_dataset`` method executes an optimized batch operation over an active, in-memory Hugging Face ``Dataset`` object. Under the hood, it leverages the high-performance ``dataset.map()`` function to apply the loaded obfuscation algorithm concurrently across rows.

.. code-block:: python

   obfuscated_dataset = tmallet.obfuscate_dataset(
       dataset=my_dataset,
       column="text",
       column_obfuscated="obfuscated_text",
       batch_size=32,
       num_proc=4
   )

Arguments
~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - **dataset**
     - ``Dataset``
     - *Required*
     - The Hugging Face dataset collection containing your raw data.
   * - **column**
     - ``str``
     - *Required*
     - The name of the column containing the raw source text.
   * - **column_obfuscated**
     - ``str``
     - *Required*
     - The name of the new target column where obfuscated results will be stored.
   * - **batch_size**
     - ``int``
     - ``10``
     - Number of examples forwarded to the obfuscator simultaneously in a single block.
   * - **num_proc**
     - ``int``
     - ``None``
     - Number of CPU cores to spawn for parallel processing. If ``None``, uses a single process.

Fault-Tolerant Chunked Streaming
---------------------------------

For massive datasets that exceed RAM capacity or are hosted directly on the Hugging Face Hub, loading everything into memory at once is impractical. The ``obfuscate_dataset_by_chunk`` method resolves this by **streaming** the dataset incrementally.

Furthermore, it implements a **checkpointing system**. As it processes each chunk, it saves the intermediate state to disk. If your process gets interrupted due to network errors, hardware timeouts, or rate limits, re-running the script will automatically skip completed chunks and resume exactly where it left off.

.. code-block:: python

   from pathlib import Path

   large_obfuscated_dataset = tmallet.obfuscate_dataset_by_chunk(
       dataset_repo="wikitext",
       column="text",
       column_obfuscated="masked_text",
       save_chunks_to_folder=Path("./checkpoints"),
       dataset_config="wikitext-103-raw-v1",
       dataset_split="train",
       chunk_size=5000
   )

Arguments
~~~~~~~~~

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Argument
     - Type
     - Default
     - Description
   * - **dataset_repo**
     - ``str``
     - *Required*
     - Hugging Face Hub repository ID (e.g., ``"imdb"``) or a local path directory.
   * - **column**
     - ``str``
     - *Required*
     - The name of the column containing the raw source text.
   * - **column_obfuscated**
     - ``str``
     - *Required*
     - The target column key where the obfuscated text is saved.
   * - **save_chunks_to_folder**
     - ``Path``
     - *Required*
     - Directory path on your local disk where chunk checkpoints will be saved/restored.
   * - **dataset_config**
     - ``str``
     - ``None``
     - Sub-dataset configuration or subset name (passed directly to ``load_dataset``).
   * - **dataset_split**
     - ``str``
     - ``"train"``
     - The data split to target (e.g., ``"train"``, ``"validation"``, ``"test"``).
   * - **chunk_size**
     - ``int``
     - ``5000``
     - The number of examples collected into a temporary block before executing obfuscation and saving a checkpoint.
   * - **batch_size**
     - ``int``
     - ``100``
     - The inner batch size sent directly to ``.map()`` within each chunk pipeline.
   * - **num_proc**
     - ``int``
     - ``None``
     - CPU core parallelism configuration split handled per chunk.
   * - **num_samples**
     - ``int``
     - ``None``
     - An optional ceiling to cap total evaluated elements (useful for quick testing runs).

.. note::
   When using chunked streaming, if a partial checkpoint folder is detected, the method prints a ``Loading checkpoint...`` status and automatically forwards the generator iterator over those records to guarantee that no data duplications or omissions occur.
