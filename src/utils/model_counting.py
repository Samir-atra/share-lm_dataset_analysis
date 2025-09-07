import datasets
import pandas as pd
from collections import Counter
import os

# This script assumes 'ours' is a loaded Hugging Face dataset.
# For standalone execution, we'll load it here.
print("Loading ShareLM dataset from Hugging Face...")
ours = datasets.load_dataset("shachardon/ShareLM")["train"]
print("Dataset loaded.")

# --- Count and analyze model names ---
print("\nAnalyzing model name distribution...")
# Use a more efficient way to count model names
model_names = [row["model_name"] for row in ours if row["model_name"]]
model_counts = Counter(model_names)

print(f"Number of rows with a valid (non-empty) model name: {len(model_names)}")

# Print the most common model counts
print("Top 10 most common models:")
for model, count in model_counts.most_common(10):
    print(f"- {model}: {count}")

# --- Filter dataset and save to a new CSV ---
print("\nFiltering dataset to include only rows with a model name...")
rows_with_model = [row for row in ours if row["model_name"]]

df_with_model = pd.DataFrame(rows_with_model)

output_csv_path = "shareLM_with_model.csv"
df_with_model.to_csv(output_csv_path, index=False)
print(f"Successfully saved {len(df_with_model)} rows to '{output_csv_path}'")