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

Here’s a structured markdown description of the algorithms and their parameters based on your provided config examples:

### **Configuration Guide**

There are four primary means of obfuscation provided:
1. Lemmatisation (somewhat obfuscates writing style; very light)
2. Scrambling (obfuscates language structure and to a lesser extent meaning; light)
3. Replacement (obfuscates word types; medium-strong)
4. Mutual Information (obfuscates based on Shannon-based metrics; adjustable).

#### **1. Lemmatisation**
**Description**: Reduces words to their base or dictionary form. A very light form of obfuscation, particularly to remove shallow elements of writing style.
To use it pass a `config` with `algorithm` set to `lemmatise`.

#### **2. Scrambling**
**Description**: Scrambling involves jumbling the words in a sentence or text. You can choose from the following scrambling algorithms:
- `scramble-BoW`: Without concern for language structure.
- `scramble-shuffle-siblings`: Parse text into a dependency tree and randomly shuffling words that are sibling nodes.
- `scramble-reverse-head`: Parse text into a dependency structure and randomly reverse the order of head nodes in relation to their siblings.

| Parameter                  | Type    | Description                                      | Default Value |
|----------------------------|---------|--------------------------------------------------|---------------|
| `scramble_within_sentence` | bool    | If `True`, scrambles words within sentences. If `False`, scrambles across the entire text. | `False`       |

#### **3. Replacement of Nouns / Proper Nouns **
**Description**: Adjusts different word types by either deleting them or replacing them with POS tags.
Algorithms
- `noun`: Keep only nouns.
- `noun-propn`: Keep only nouns and proper nouns.
- `no-noun`: Keep everything but nouns.
- `no-noun-propn`: Keep everything but nouns and proper nouns.

Additional Configuration Options:
| Parameter            | Type    | Description                                      | Default Value |
|----------------------|---------|--------------------------------------------------|---------------|
| `replace_with_pos`   | bool    | If `True`, replaces with the specified POS.      | `True`        |

#### **4. Mutual Information Obfuscation **
**Description**: Applies Shannon entropy-based text transformation, replacing words based on a threshold of Mutual Information. To use this pass `shannon` to the `algorithm` parameter in the configuration.

Additional Configuration Options:
| Parameter      | Type    | Description                                      | Default Value |
|----------------|---------|--------------------------------------------------|---------------|
| `threshold`    | int     | Threshold for character replacement.             | `10`          |
| `replace_with` | str     | Character used for replacement.                  | `"_"`         |

#### Acknowledgements
Part of this work was conducted within the [CORAL project](https://coral-nlp.github.io) funded by the German Federal Ministry of Research, Technology, and Space (BMFTR) under the grant number 16IS24077A. Responsibility for the content of this publication lies with the authors. 
