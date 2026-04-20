[![Powered by spaCy](https://img.shields.io/badge/Powered%20by-spaCy-09a3d5?logo=spacy&logoColor=white)](https://spacy.io)
[![English]](https://img.shields.io/badge/lang-en-blue)
[![German]](https://img.shields.io/badge/lang-de-blue)

<br />
<div align="center">
  <img src="assets/mallet.svg" alt="Logo" width="200" height="200">

  <p align="center">
        Smash Text Into Obfuscated Formats
    <br />
    <br />
    <br />
    <a href="https://text-mallet.readthedocs.io/en/latest/">Read The Docs</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/DanielGall500/text-hammer/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

A package for the smashing of text into [derived](https://text-plus.org/en/themen-dokumentation/atf) formats, aimed at reducing the possibility of privacy or copyright infringement while maintaining the text utility for certain tasks e.g. classification, retrieval.
Supports obfuscation of text in English or German.

When we think about how we might be able to transform text, we can look at the following aspects:
* Word Forms (the character sequence)
* Root Forms (lemmas)
* Syntactic and Morpho-Syntactic Features
* Meanings
* Grammatical Relations (hierarchical structure)
* Sequence Information (linear structure)

Each of the above contributes a certain amount of *information* to the final text. This tool allows you to directly or indirectly erode such information.
Languages vary significantly in which they most rely on for certain features, for instance English relies heavily on structure for assigning grammatical case while German relies more on morphological adjustments with relatively free word order.

#### Why obfuscate text?
When training models for text generation, we typically need all of the content and style of the original, fluent text. However, there are many tasks such as classification, semantic similarity scoring, topic modelling, and so on, where the original text may not be required in its original form to help model performance. There is typically a trove of public-domain data that can be used for model training, but there are still many questions around the usage of copyright-protected data in training. This package offers a route to preserve some of the value of copyrighted texts while hindering their reconstruction, whether that be through training-data reconstruction or model outputs.

The creation of transformed texts that are thus no longer consumable by humans, but still useful for training on specific tasks.

#### Won't that reduce model performance?
In many cases, this approach will likely reduce model performance if we compare it to a model trained on the original data.
However, the goal is to _add_ these additional obfuscated text formats to our original, public-domain text data. 
This approach ensures that the original copyright-protected text never enters the model in any consumable format, thus offering strong protection again copyright infringement.

## Usage

### Basic Obfuscation
```python
from tmallet import TMallet

config = {
    "algorithm": "retain-noun-propn", 
    "replacement_mechanism": "POS"
}

tmallet = TMallet(lang="en")
tmallet.load_obfuscator(config)

text = """
A Soyuz rocket launched two Galileo satellites into orbit on Friday,
marking a crucial step for Europe’s planned navigation system,
operator Arianespace announced.
"""
obfuscated_text = mallet.obfuscate(text, config)
```
Output
```bash
"DET Soyuz rocket VERB NUM Galileo satellites ADP orbit ADP Friday PUNCT
VERB DET ADJ step ADP Europe PART VERB navigation system PUNCT
operator Arianespace VERB "
```

**Obfuscate based on an approximation of 'word importance'**
Mutual information measures how much information context tells you about a word.
Words which are both _rare_ and _context-dependant_ tend to be _important_ to the meaning of a text.
We can apply a filter to set upper or lower bounds on such an MI score, filering at the word level.
```python
from tmallet import TMallet

config = {
    "algorithm": "shannon", 
    "threshold": 8,
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": "DEFAULT"
}

tmallet = TMallet(lang="en")
tmallet.load_obfuscator(config)

...
text = """
Three-dimensional printing is being used to make metal parts 
for aircraft and space vehicles.
"""
obfuscated_text = mallet.obfuscate(text)
```

Output
```
==Mutual-Information Obfuscation==
Threshold:  8
Lower Bounded:  Three _ dimensional printing _ _ used _ _ metal parts _ aircraft _ space vehicles _
Upper Bounded:  _ - _ _ is being _ to make _ _ for _ and _ _.
==================================
```

### Obfuscate a Dataset
```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_huggingface_dataset")

# Obfuscate a column
config = {"algorithm": "lemmatize"}

tmallet = TMallet()
tmallet.load_obfuscator(config)

obfuscated_dataset = tmallet.obfuscate_dataset(
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
...

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
2. Scrambling (obfuscates language structure; light-to-medium)
3. Part-of-Speech Filtering (obfuscates word types; medium-strong)
4. Mutual-Information Filtering (obfuscates based on an approximation of 'word importance' in a text; adjustable).

#### 1. Lemmatisation
**Description**: Reduces words to their base or dictionary form. A very light form of obfuscation, particularly to remove shallow elements of writing style.
To use it pass a `config` with `algorithm` set to `lemmatize`.

#### 2. Scrambling
**Description**: Scrambling involves jumbling the words in a sentence or text. You can choose from the following scrambling algorithms:
- `scramble-hier-weak`: Use dependency parsing to scramble words (child nodes i.e. words follow their head) that of the same head and the same side of said head.
- `scramble-hier-strong`: Use dependency parsing to scramble words (child nodes i.e. words follow their head) of the same head. 
- `scramble-BoW-sentence`: Randomly scramble words (within sentences) without concern for language structure.
- `scramble-BoW-document`: Randomly scramble words (across the entire text) without concern for language structure.

#### 3. Replacement of Nouns / Proper Nouns
**Description**: Adjusts different word types by either deleting them or replacing them with POS tags.
Algorithms
- `retain-noun`: Keep only nouns.
- `retain-noun-propn`: Keep only nouns and proper nouns.
- `remove-noun`: Keep everything but nouns.
- `remove-noun-propn`: Keep everything but nouns and proper nouns.

Additional Configuration Options:
| Parameter            | Type    | Description                                      | Default Value |
|----------------------|---------|--------------------------------------------------|---------------|
| `replacement_mechanism`   | str    | Determines how filtered words are replaced in the text. Can be one of "DEFAULT" (replaced with a default character, typically an underscore), "DELETE", or "POS" (replaced with the corresponding part-of-speech tag).     | `DEFAULT`        |

#### 4. Mutual Information Obfuscation
**Description**: Applies Shannon entropy-based text transformation, replacing words based on a threshold of Mutual Information. To use this pass `shannon` to the `algorithm` parameter in the configuration.

Additional Configuration Options:
| Parameter      | Type    | Description                                      | Default Value |
|----------------|---------|--------------------------------------------------|---------------|
| `threshold`    | int     | Threshold for character replacement.             | `10`          |
| `replacement_mechanism`   | str    | Determines how filtered words are replaced in the text. Can be one of "DEFAULT" (replaced with a default character, typically an underscore), "DELETE", or "POS" (replaced with the corresponding part-of-speech tag).     | `DEFAULT`        |
| `as_upper_bound`    | bool     | Whether all words with a MI value above the threshold should be filtered.             | `True`          |
| `as_lower_bound`    | bool     | Whether all words with a MI value below the threshold should be filtered.             | `True`          |
| `output_mi_values`    | bool     | Whether the MI values themselves should be provided.             | `False`          |

#### Acknowledgements
Part of this work was conducted within the [CORAL project](https://coral-nlp.github.io) funded by the German Federal Ministry of Research, Technology, and Space (BMFTR) under the grant number 16IS24077A. Responsibility for the content of this publication lies with the authors. 
