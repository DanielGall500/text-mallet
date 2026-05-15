from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, DownloadMode
import random
import re

from datasets.arrow_dataset import sys

# -----------------------------
# Load dataset from disk
# -----------------------------
dataset_path_train = "DanielGallagherIRE/fineweb-edu-1B-obfuscated"
dataset_train = load_dataset(dataset_path_train, name="shannon", split="train") 
df_train = dataset_train.to_pandas()
print(df_train.head())
print(df_train.columns)
print(len(df_train))

import sys
sys.exit()

dataset_path = "experiments/merged-obfuscation/"
dataset = load_from_disk(dataset_path)

data = dataset

df = data.to_pandas()

# -----------------------------
# Filter columns (ending in digit)
# -----------------------------
pattern = re.compile(r"\d$")  # ends with a digit
cols_to_keep = [col for col in data.column_names if pattern.search(col) or col == "text"]

df = df[cols_to_keep]

print("\n🧾 Pandas DataFrame Preview")
print("-------------------")
print(df.head())

print("Columns by the end:", df.columns)

# -----------------------------
# Convert DataFrame → Dataset
# -----------------------------
mi_dataset = Dataset.from_pandas(df, preserve_index=False)
print(mi_dataset)
print(df.columns)
print(len(df))

# -----------------------------
# Create DatasetDict with new split
# -----------------------------
dataset_dict = DatasetDict({
    "shannon": mi_dataset
})

print("\n📦 New DatasetDict")
print(dataset_dict)

import sys
sys.exit()
# -----------------------------
# Push to Hugging Face Hub
# -----------------------------
repo_id = "DanielGallagherIRE/fineweb-edu-1B-obfuscated"
dataset_dict.push_to_hub(repo_id)

print(f"\n🚀 Uploaded new split 'mi' to: https://huggingface.co/datasets/{repo_id}")
