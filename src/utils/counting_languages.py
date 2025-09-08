"""
Language Counting Script

This script reads the 'sharelm_with_model.csv' dataset, extracts the language
from the 'conversation_metadata' for each conversation, and prints a summary
of the language counts.

Usage:
    python src/visualizations/counting_languages.py
"""
import pandas as pd
import json
from collections import Counter
import os

def parse_metadata(metadata_str):
    """
    Safely parses a string that represents a dictionary or JSON object.
    """
    if not isinstance(metadata_str, str):
        return {}  # Return empty dict if it's not a string (e.g., NaN)
    try:
        # Replace single quotes with double quotes for valid JSON
        return json.loads(metadata_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        # Return an empty dictionary if parsing fails
        return {}

def count_languages(file_path):
    """
    Reads a CSV file, extracts languages from the 'conversation_metadata' column,
    and prints the count for each language.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Reading data from '{file_path}'...")
    # Read the CSV, treating the metadata column as a string for safe parsing.
    df = pd.read_csv(file_path, dtype={'conversation_metadata': 'object'}, low_memory=False)

    # Create a 'language' series, defaulting to 'N/A' if not found.
    languages = df['conversation_metadata'].apply(parse_metadata).apply(lambda x: x.get('language', 'N/A'))

    # Use Counter to get the frequency of each language.
    language_counts = Counter(languages)

    print("\n--- Language Counts ---")
    print(f"Total unique languages found: {len(language_counts)}")
    print(f"Total conversations analyzed: {len(df)}")
    print("-" * 25)
    # Print the most common languages first.
    for language, count in language_counts.most_common():
        print(f"- {language}: {count}")

if __name__ == "__main__":
    csv_file_path = "sharelm_with_model.csv"
    count_languages(csv_file_path)