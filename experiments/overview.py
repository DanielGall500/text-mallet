from datasets import load_from_disk
import random
import re

# -----------------------------
# Load dataset from disk
# -----------------------------
dataset_path = "experiments/merged-obfuscation/"
dataset = load_from_disk(dataset_path)

data = dataset

df = data.to_pandas()
features = df.columns

pattern = re.compile(r"\d$")  # ends with a digit
cols_to_keep = [col for col in data.column_names if pattern.search(col)]

df = df[cols_to_keep]

print("\n🧾 Pandas DataFrame Preview")
print("-------------------")
print(df.head())

print("Columns by the end: ", df.columns)

# -----------------------------
# Dataset overview
# -----------------------------
print("\n📊 Dataset Overview")
print("-------------------")
print(f"Number of rows: {len(data)}")
print(f"Columns: {data.column_names}")
print(f"Features:\n{data.features}")


# -----------------------------
# Show random samples
# -----------------------------
print("\n🎲 Random samples")
print("-------------------")
for _ in range(min(3, len(data))):
    idx = random.randint(0, len(data) - 1)
    print(f"\nRandom Row {idx}:")
    print(data[idx])
