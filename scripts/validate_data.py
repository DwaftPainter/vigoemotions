
import pandas as pd
from pathlib import Path

data_dir = Path("data")
splits = ["train", "val", "test"]

datasets = {}

print("=== DATASET VALIDATION ===")

for split in splits:
    file_path = data_dir / f"{split}.csv"

    df = pd.read_csv(file_path)
    datasets[split] = df

    print(f"\n=== {split.upper()} ===")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Missing values:")
    print(df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

# Combine splits for dataset-level checks
full_df = pd.concat(
    datasets.values(),
    ignore_index=True
)

print("\n=== FULL DATASET ===")
print("Total samples:", len(full_df))
print("Expected samples:", 20664)

print("Total matches expected:",
      len(full_df) == 20664)

print("Duplicate IDs:",
      full_df.duplicated(subset=["id"]).sum())

print("Duplicate texts:",
      full_df.duplicated(subset=["text"]).sum())

print("\n=== SCHEMA ===")
print(full_df.dtypes)

print("\n=== MISSING VALUES ===")
print(full_df.isnull().sum())


print("\n=== DUPLICATE TEXT ANALYSIS ===")

parts = []

for split, df in datasets.items():
    temp = df.copy()
    temp["split"] = split
    parts.append(temp)

combined = pd.concat(parts, ignore_index=True)

duplicates = combined[
    combined.duplicated(subset=["text"], keep=False)
]

# Không in nội dung comment thô
print(duplicates[["id", "split", "labels"]].to_string(index=False))

cross_split = (
    duplicates.groupby("text")["split"]
    .nunique()
    .gt(1)
    .sum()
)

print("Texts duplicated across splits:", cross_split)
