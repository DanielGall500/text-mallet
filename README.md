[![Powered by spaCy](https://img.shields.io/badge/Powered%20by-spaCy-09a3d5?logo=spacy&logoColor=white)](https://spacy.io)
![English](https://img.shields.io/badge/lang-en-blue)
![German](https://img.shields.io/badge/lang-de-blue)

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
There are multiple general obfuscation approaches to choose from, separated into three general categories:
* Structural Obfuscation 
* Part-of-Speech Filtering 
* Mutual-Information Filtering

Let's start with an example of POS filtering, where we retain (`filter_type: 'retain'`) any common and proper nouns (`pos_tags: ["NOUN","PROPN"]`).
```python
from tmallet import TMallet

sample_text = "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe.", 

# Choose your obfuscation algorithm and configuration.
algorithm = "pos-filter"
config = {
    "filter_type": "retain",
    "pos_tags": ["NOUN","PROPN"],
    "replacement_mechanism": "DEFAULT",
    "seed": 100,
}

# Load the TMallet Obfuscator for a language, algorithm, and configuration.
tmallet = TMallet(lang=lang, prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)

# Obfuscate!
obfuscated = tmallet.obfuscate(sample)
obfuscated_text = obfuscated["pos-filter"]["retain"]["POS"]
print(obfuscated_text)
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

algorithm = "shannon", 
config = {
    "threshold": 8,
    "as_upper_bound": True,
    "as_lower_bound": True,
    "replacement_mechanism": "DEFAULT"
}

tmallet = TMallet(lang=lang, prefer_gpu=True)
tmallet.load_obfuscator(algorithm, config)

text = "Leipzig is the most populous city in the German state of Saxony. The city has a population of 633,592 residents as of 31 December 2025. It is the eighth-largest city in Germany and is part of the Central German Metropolitan Region. Leipzig is located about 150 km (90 mi) southwest of Berlin, in the southernmost part of the North German Plain (the Leipzig Bay), at the confluence of the White Elster and its tributaries Pleiße and Parthe."
obfuscated = tmallet.obfuscate(text)
print(obfuscated)
```

Output
```json
{
  "7.5": {
    "as_upper_bound": {
      "DELETE": "is the in the of . The a of, as of . It is the - in and is of the . is () of, in the of  the (the Bay), at the of the White and and.",
      "DEFAULT": "_ is the _ _ _ in the _ _ of _ . The _ _ a _ of _, _ _ as of _ _ _ . It is the _ - _ _  in _ and is _ of the _ _ _ _ . _ is _ _ _ _ (_ _) _ of _, in the _ _ of the _ _ _ (the _ Bay), at the _ of the Wh ite _ and _ _ _ and _.",
      "POS": "PROPN is the ADV ADJ NOUN in the ADJ NOUN of PROPN . The NOUN AUX a NOUN of NUM, NUM NOUN  as of NUM PROPN NUM . It is the ADV - ADJ NOUN in PROPN and is NOUN of the ADJ ADJ PROPN PROPN . PROPN is VERB ADP  NUM NOUN (NUM NOUN) ADV of PROPN, in the ADJ NOUN of the ADJ ADJ PROPN (the PROPN Bay), at the NOUN of the White  PROPN and PRON NOUN PROPN and PROPN."
    },
    "as_lower_bound": {
      "DELETE": "Leipzig most populous city German state Saxony city has population 633 592 residents 31  December 2025 eighth largest city Germany part Central German Metropolitan Region Leipzig located about 150 km 90  mi southwest Berlin southernmost part North German Plain Leipzig confluence Elster its tributaries Pleiße Pa rthe",
      "DEFAULT": "Leipzig _ _ most populous city _ _ German state _ Saxony _ _ city has _ population _ 6 33 _ 592 residents _ _ 31 December 2025 _ _ _ _ eighth _ largest city _ Germany _ _ part _ _ Central German Metrop olitan Region _ Leipzig _ located about 150 km _ 90 mi _ southwest _ Berlin _ _ _ southernmost part _ _ North Germ an Plain _ _ Leipzig _ _ _ _ _ confluence _ _ _ Elster _ its tributaries Pleiße _ Parthe _",
      "POS": "Leipzig AUX DET most populous city ADP DET German state ADP Saxony PUNCT DET city has DET  population ADP 633 PUNCT 592 residents ADP ADP 31 December 2025 PUNCT PRON AUX DET eighth PUNCT largest city ADP G ermany CCONJ AUX part ADP DET Central German Metropolitan Region PUNCT Leipzig AUX located about 150 km PUNCT 90 m i PUNCT southwest ADP Berlin PUNCT ADP DET southernmost part ADP DET North German Plain PUNCT DET Leipzig PROPN PU NCT PUNCT ADP DET confluence ADP DET PROPN Elster CCONJ its tributaries Pleiße CCONJ Parthe PUNCT"
    }
  }
}
```

If we obfuscate too strongly using mutual information, we'll end up with obfuscated sentences like:
```
, . . - ., or . the / . the . - –, . -
```
That's, well, probably not very useful. Ideally, we can find a balance between the obfuscation of some words and inclusion of others.
Here's an overview of an approximation of pointwise word-level mutual information, i.e. PMI(word; context), over 12,000 tokens taken from 10 random texts in the FineWeb-Edu dataset, for instance.
![Distribution of Word-Level Mutual Information](assets/mutual-info-distribution.png)

### Obfuscate a Dataset
```python
from datasets import load_dataset

# 1. Load in a dataset from HF
dataset = load_dataset("your_huggingface_dataset")

# 2. Set up the text-mallet obfuscator as above
...

# 3. Perform obfuscation over an entire dataset,
obfuscated_dataset = tmallet.obfuscate_dataset(
    dataset=dataset,
    column="text",
    column_obfuscated="text_obfuscated",
    config=config,
    batch_size=100
)
```

Alternatively, you can also use chunking to save progress as you obfuscate:
```python
...

obfuscated_dataset = tmallet.obfuscate_dataset_by_chunk(
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
To use it simply set `algorithm` set to `lemmatize`. A configuration is not required.

#### 2. Scrambling
**Description**: Scrambling involves jumbling the words in a sentence or text. You can choose from the following scrambling algorithms:
- `scramble-hier`
    - `weak` Use dependency parsing to scramble words (child nodes i.e. words follow their head) that of the same head and the same side of said head.
    - `strong`: Use dependency parsing to scramble words (child nodes i.e. words follow their head) of the same head. 

- `scramble-BoW`
    - `sentence`: Randomly scramble words (within sentences) without concern for language structure.
    - `document`: Randomly scramble words (across the entire text) without concern for language structure.

#### 3. Replacement of Nouns / Proper Nouns
**Description**: Adjusts different word types by either deleting them or replacing them with POS tags.
Algorithm: `pos-filter`

Configuration Options:
| Parameter            | Type    | Description                                      | Default Value |
|----------------------|---------|--------------------------------------------------|---------------|
| `filter_type`   | str or List[str]   | Choose whether targeted POS tags are removed or retained.     | `"retain"`        |
| `pos_tags`   | str or List[str]   | Choose the POS tags targeted in the filtering.     | `["NOUN","PROPN"]`        |
| `replacement_mechanism`   | str    | Determines how filtered words are replaced in the text. Can be one of "DEFAULT" (replaced with a default character, typically an underscore), "DELETE", or "POS" (replaced with the corresponding part-of-speech tag).     | `DEFAULT`        |

#### 4. Mutual Information Obfuscation
**Description**: Applies Shannon entropy-based text transformation, replacing words based on a threshold of Mutual Information. To use this pass `shannon` to the `algorithm` parameter in the configuration.
Algorithm: `shannon`

Additional Configuration Options:
| Parameter      | Type    | Description                                      | Default Value |
|----------------|---------|--------------------------------------------------|---------------|
| `threshold`    | int     | Threshold for character replacement.             | `10`          |
| `replacement_mechanism`   | str    | Determines how filtered words are replaced in the text. Can be one of "DEFAULT" (replaced with a default character, typically an underscore), "DELETE", or "POS" (replaced with the corresponding part-of-speech tag).     | `DEFAULT`        |
| `as_upper_bound`    | bool     | Whether all words with a MI value above the threshold should be filtered.             | `True`          |
| `as_lower_bound`    | bool     | Whether all words with a MI value below the threshold should be filtered.             | `True`          |
| `max_context_length`    | int     | Maximum context length for the BERT model used.             | `8192`          |
| `output_mi_values`    | bool     | Whether the MI values themselves should be provided.             | `False`          |

#### Acknowledgements
Part of this work was conducted within the [CORAL project](https://coral-nlp.github.io) funded by the German Federal Ministry of Research, Technology, and Space (BMFTR) under the grant number 16IS24077A. Responsibility for the content of this publication lies with the authors. 
