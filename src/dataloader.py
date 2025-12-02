from datasets import load_dataset, Dataset, DatasetDict
from typing import Callable, Optional, Union, List
import pandas as pd
from pathlib import Path
import spacy


class DatasetLoader:
    """
    A class to load HuggingFace datasets and apply various alterations.
    """

    def __init__(self, dataset_name: str, split: Optional[str] = None, **kwargs):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to load (e.g., 'train', 'test'). If None, loads all splits
            **kwargs: Additional arguments to pass to load_dataset()
        """
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name, split=split, **kwargs)
        self.original_dataset = self.dataset

    def filter(self, condition: Callable, batched: bool = False, **kwargs):
        """
        Filter the dataset based on a condition.

        Args:
            condition: Function that returns True for examples to keep
            batched: Whether to process examples in batches
            **kwargs: Additional arguments for the filter operation
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.filter(condition, batched=batched, **kwargs)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.filter(condition, batched=batched, **kwargs)
        return self

    def map(self, function: Callable, batched: bool = False, remove_columns: Optional[List[str]] = None, **kwargs):
        """
        Apply a function to transform the dataset.

        Args:
            function: Function to apply to each example
            batched: Whether to process examples in batches
            remove_columns: Columns to remove after mapping
            **kwargs: Additional arguments for the map operation
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.map(function, batched=batched, remove_columns=remove_columns, **kwargs)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.map(function, batched=batched, remove_columns=remove_columns, **kwargs)
        return self

    def select(self, indices: Union[List[int], range]):
        """
        Select specific examples by index.

        Args:
            indices: List or range of indices to select
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.select(indices)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.select(indices)
        return self

    def shuffle(self, seed: Optional[int] = None):
        """
        Shuffle the dataset.

        Args:
            seed: Random seed for reproducibility
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.shuffle(seed=seed)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        return self

    def rename_column(self, original: str, new: str):
        """
        Rename a column in the dataset.

        Args:
            original: Original column name
            new: New column name
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.rename_column(original, new)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.rename_column(original, new)
        return self

    def remove_columns(self, columns: Union[str, List[str]]):
        """
        Remove columns from the dataset.

        Args:
            columns: Column name or list of column names to remove
        """
        if isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({
                split: ds.remove_columns(columns)
                for split, ds in self.dataset.items()
            })
        else:
            self.dataset = self.dataset.remove_columns(columns)
        return self

    def train_test_split(self, test_size: float = 0.2, seed: Optional[int] = None):
        """
        Split the dataset into train and test sets.

        Args:
            test_size: Proportion of dataset to use for testing
            seed: Random seed for reproducibility
        """
        if isinstance(self.dataset, DatasetDict):
            print("Dataset already has splits. Skipping train_test_split.")
        else:
            self.dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        return self

    def reset(self):
        """Reset to the original dataset."""
        self.dataset = self.original_dataset
        return self

    def get_dataset(self) -> Union[Dataset, DatasetDict]:
        """Return the current dataset."""
        return self.dataset

    def to_pandas(self, split: Optional[str] = None) -> Union[pd.DataFrame, dict]:
        """
        Convert dataset to pandas DataFrame.

        Args:
            split: Specific split to convert (if DatasetDict). If None, converts all splits
        """
        if isinstance(self.dataset, DatasetDict):
            if split:
                return self.dataset[split].to_pandas()
            return {split: ds.to_pandas() for split, ds in self.dataset.items()}
        return self.dataset.to_pandas()

    def info(self):
        """Print information about the dataset."""
        if isinstance(self.dataset, DatasetDict):
            for split, ds in self.dataset.items():
                print(f"\n{split.upper()} split:")
                print(f"  Rows: {len(ds)}")
                print(f"  Columns: {ds.column_names}")
        else:
            print(f"Rows: {len(self.dataset)}")
            print(f"Columns: {self.dataset.column_names}")


def to_nouns(s: str, nlp) -> str:
    doc = nlp(s)
    nouns = []
    for token in doc:
        is_noun = token.pos_ == 'NOUN' or token.pos_ == 'PROPN'
        if is_noun:
            nouns.append(token.text)
    return " ".join(nouns)

if __name__ == "__main__":
    # Configuration
    FRAGMENT_SIZE = 10000  # Process 10k rows at a time
    OUTPUT_FILE = "datasets/fineweb-edu-1B-nouns.csv"

    # Load spaCy model once
    nlp = spacy.load("en_core_web_sm")
    spacy.require_gpu()
    print(f"Using GPU: {spacy.prefer_gpu()}")

    # Load dataset
    loader = DatasetLoader("codelion/fineweb-edu-1B", split="train")
    print(loader.info())

    total_rows = len(loader.dataset)
    print(f"Total rows: {total_rows:,}")

    # Create output directory if needed
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Process in fragments
    first_fragment = True
    for start_idx in range(0, total_rows, FRAGMENT_SIZE):
        end_idx = min(start_idx + FRAGMENT_SIZE, total_rows)
        print(f"Processing rows {start_idx:,} to {end_idx:,}...")

        # Select fragment
        fragment = loader.dataset.select(range(start_idx, end_idx))
        fragment_df = fragment.to_pandas()

        # Extract texts
        texts = fragment_df["text"]

        # Process with spaCy
        texts_as_nouns = []
        for i, doc in enumerate(nlp.pipe(texts, batch_size=32)):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(texts)} texts in fragment")
            nouns = " ".join([token.text for token in doc if token.pos_ == "NOUN"])
            texts_as_nouns.append(nouns)

        # Add processed column
        fragment_df["text_nouns"] = texts_as_nouns

        # Write to CSV (append mode after first fragment)
        if first_fragment:
            fragment_df.to_csv(OUTPUT_FILE, index=False, mode='w')
            first_fragment = False
        else:
            fragment_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        print(f"Saved fragment to {OUTPUT_FILE}")

        # Clear memory
        del fragment, fragment_df, texts, texts_as_nouns

    print(f"\nSaved to {OUTPUT_FILE}")
