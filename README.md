[![Powered by spaCy](https://img.shields.io/badge/Powered%20by-spaCy-09a3d5?logo=spacy&logoColor=white)](https://spacy.io)

<br />
<div align="center">
  <img src="assets/mallet.svg" alt="Logo" width="200" height="200">

  <p align="center">
        Smash Text Into Obfuscated Formats
    <br />
    <br />
    <br />
    <a href="">View Demo</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

A package for the smashing of text into [derived](https://text-plus.org/en/themen-dokumentation/atf) formats, aimed at reducing the possibility of privacy or copyright infringement while maintaining the text utility for certain tasks e.g. classification, retrieval.

When we think about how strings can be altered for obfuscation, we can look at the following aspects:
* Word Forms (the character sequence)
* Root Forms (lemmas)
* Syntactic and Morpho-Syntactic Features
* Meanings
* Grammatical Relations (hierarchical structure)
* Sequence Information (linear structure)

Each of the above contributes a certain amount of *information* to the final text. This tool allows you to directly or indirectly reduce the information present in a text.
Languages vary significantly in which they most rely on for certain features, for instance English relies heavily on structure for assigning grammatical case while German relies more on morphological adjustments with relatively free word order.

## Usage

### Basic Obfuscation
```python
from tmallet import TMallet

# Initialise the obfuscator
mallet = TMallet()

# Transform a text into Nouns, Proper Nouns, and POS tags
text = """
A Soyuz rocket launched two Galileo satellites into orbit on Friday,
marking a crucial step for Europe’s planned navigation system,
operator Arianespace announced.
"""
config = {"algorithm": "noun-propn-pos"}
obfuscated_text = mallet.obfuscate(text, config)
```
Output
```bash
"DET Soyuz rocket VERB NUM Galileo satellites ADP orbit ADP Friday PUNCT
VERB DET ADJ step ADP Europe PART VERB navigation system PUNCT
operator Arianespace VERB "
```

### Obfuscate a Dataset
```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_huggingface_dataset")

# Obfuscate a column
mallet = TMallet()
config = {"algorithm": "lemmatization"}

obfuscated_dataset = mallet.obfuscate_dataset(
    dataset=dataset,
    column="text",
    column_obfuscated="text_obfuscated",
    config=config,
    batch_size=100
)
```

### Obfuscate Large Datasets with Checkpointing

For large datasets, use chunked processing to save progress:
```python
from pathlib import Path

mallet = TMallet()
config = {"algorithm": "scramble-BoW"}

obfuscated_dataset = mallet.obfuscate_dataset_by_chunk(
    dataset=dataset,
    column="text",
    column_obfuscated="text_obfuscated",
    config=config,
    save_chunks_to_folder=Path("./checkpoints"),
    chunk_size=5000,
    batch_size=100
)
```

### Available Algorithms

- `"noun"`, `"noun-propn"`, `"noun-pos"`, `"noun-propn-pos"`
- `"lemmatization"`
- `"scramble-BoW"`, `"scramble-BoW-by-sentence"`
- `"scramble-shuffle-siblings"`, `"scramble-reverse-head"`
- `"mutual-information"`
