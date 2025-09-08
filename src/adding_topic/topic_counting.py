"""
Topic Analysis and Cleaning Script

This script performs two main functions on the 'sharelm_with_model.csv' dataset:
1.  **Topic Counting (Default Behavior):** It reads the dataset, parses the
    'conversation_metadata' column to extract the 'topic' for each row,
    and prints a summary of the topic distribution. This includes the total
    number of unique topics and the count for each one.

2.  **Interactive Cleaning (Optional):** When run with the `--clean` flag,
    the script will first perform the topic count. Then, it identifies rows
    where the topic is 'Error: API call failed again' or 'No Topic Found'
    (which includes empty topics). It will report the number of such rows
    and prompt the user for confirmation to remove them. If confirmed, it
    overwrites the original CSV file with the cleaned data.

Usage:
    - To count topics only:
      python src/adding_topic/topic_counting.py

    - To count topics and get an option to clean the data:
      python src/adding_topic/topic_counting.py --clean
"""
import pandas as pd
import json
from collections import Counter
import os
import argparse

# Specify the path to your CSV file
csv_file_path = "sharelm_with_model.csv"

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

def count_topics(file_path):
    """
    Reads a CSV file, extracts topics from the 'conversation_metadata' column,
    and prints the count for each topic.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Reading data from '{file_path}'...")
    # Read the CSV, treating the metadata column as a string
    df = pd.read_csv(file_path, dtype={'conversation_metadata': 'object'}, low_memory=False)

    # Create a 'topic' column for easier processing
    df['topic'] = df['conversation_metadata'].apply(parse_metadata).apply(lambda x: x.get('topic'))

    # Use Counter to count the occurrences of each topic
    topic_counts = Counter(df['topic'])

    # Calculate the total number of rows that have a topic (i.e., not None)
    total_with_topic = sum(count for topic, count in topic_counts.items() if topic is not None)

    print("\n--- Topic Counts ---")
    print(f"Total unique topics found: {len(topic_counts)}")
    print(f"Total conversations with a topic: {total_with_topic} out of {len(df)} rows")
    for topic, count in topic_counts.most_common():
        print(f"- {topic if topic is not None else 'No Topic Found'}: {count}")
    
    return df

def clean_topics(df, file_path):
    """
    Identifies rows with problematic topics, prompts the user for removal,
    and saves the cleaned DataFrame if confirmed.
    """
    topics_to_remove = ["Error: API call failed again", "No Topic Found"]
    
    # Create a boolean mask for rows to be removed
    # We also include rows where the topic is None, which corresponds to 'No Topic Found' in the counter
    mask_to_remove = df['topic'].isin(topics_to_remove) | df['topic'].isnull()
    
    rows_to_remove = df[mask_to_remove]
    
    if rows_to_remove.empty:
        print("\nNo rows found with topics 'Error: API call failed again' or 'No Topic Found'.")
        return

    print(f"\nFound {len(rows_to_remove)} rows with topics to remove: {topics_to_remove}")
    
    user_input = input("Do you want to remove these rows from the dataset and save the changes? (y/n): ").lower()
    
    if user_input == 'y':
        df_cleaned = df[~mask_to_remove].drop(columns=['topic']) # Drop the temporary 'topic' column
        df_cleaned.to_csv(file_path, index=False)
        print(f"Removed {len(rows_to_remove)} rows. The cleaned dataset has been saved to '{file_path}'.")
    else:
        print("No changes were made to the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count topics and optionally clean the dataset.")
    parser.add_argument('--clean', action='store_true', help='Enable interactive cleaning of rows with specific error topics.')
    args = parser.parse_args()

    df_with_topics = count_topics(csv_file_path)
    if args.clean:
        clean_topics(df_with_topics, csv_file_path)