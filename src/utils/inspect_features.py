#!/usr/bin/env python3
"""
Feature Inspection Script

This script loads the ShareLM dataset and inspects the unique values
for a specified set of categorical features. It's useful for understanding
the possible values each feature can take.

Usage:
    python src/utils/inspect_features.py
"""
import pandas as pd
import json

def parse_metadata(metadata_str):
    """
    Safely parses a string that represents a dictionary or JSON object.
    Handles single quotes and other common formatting issues.
    """
    if not isinstance(metadata_str, str):
        return {}  # Return empty dict if it's not a string (e.g., NaN)
    try:
        # Replace single quotes with double quotes for valid JSON
        return json.loads(metadata_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        # Return an empty dictionary if parsing fails
        return {}

def inspect_feature_values(df, features_to_inspect):
    """
    Prints the unique values for each specified feature in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_to_inspect (list): A list of column names to inspect.
    """
    if 'conversation_metadata' not in df.columns:
        print("Error: 'conversation_metadata' column not found in the dataset.")
        return

    # Parse the metadata column first
    metadata_series = df['conversation_metadata'].apply(parse_metadata)

    for feature in features_to_inspect:
        # Extract the feature from each metadata dictionary
        feature_values = metadata_series.apply(lambda x: x.get(feature, 'Not Found'))
        unique_values = feature_values.unique()
        print(f"\n--- Unique values for '{feature}' in conversation_metadata ---")
        for value in unique_values:
            print(f"- {value}")

if __name__ == '__main__':
    try:
        file_path = 'sharelm_with_model.csv'
        print(f"Loading data from '{file_path}'...")
        df = pd.read_csv(file_path, dtype={'conversation_metadata': 'object'}, low_memory=False)
        print("Data loaded successfully.")

        features = ['custom_instruction', 'status', 'Redacted']
        inspect_feature_values(df, features)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")