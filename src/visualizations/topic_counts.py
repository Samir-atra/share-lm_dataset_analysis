import pandas as pd
import json
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Create a 'topic' series for easier processing, replacing None with a string
    topics = df['conversation_metadata'].apply(parse_metadata).apply(lambda x: x.get('topic', 'No Topic Found'))

    # Use Counter to count the occurrences of each topic
    topic_counts = Counter(topics)

    # Calculate the total number of rows that have a topic (i.e., not None)
    total_with_topic = sum(count for topic, count in topic_counts.items() if topic != 'No Topic Found')

    print("\n--- Topic Counts ---")
    print(f"Total unique topics found: {len(topic_counts)}")
    print(f"Total conversations with a topic: {total_with_topic} out of {len(df)} rows")
    for topic, count in topic_counts.most_common():
        print(f"- {topic}: {count}")
    
    return topic_counts

def plot_topics(topic_counts):
    """
    Generates and displays a bar chart for topic counts.
    """
    # Convert Counter to a pandas Series for easier plotting
    topic_counts_series = pd.Series(topic_counts).sort_values(ascending=False)

    # Create a single figure for the plot
    plt.figure(figsize=(12, 8))

    # Create a horizontal bar chart of topic counts
    ax = topic_counts_series.plot(kind='barh')
    plt.title('Frequency of Topics in ShareLM Dataset')
    plt.xlabel('Count')
    plt.ylabel('Topic')
    plt.gca().invert_yaxis() # Show most frequent at the top

    # Add the exact count on each bar
    for index, value in enumerate(topic_counts_series):
        # The y-coordinate is the index, and the x-coordinate is the value (count)
        # We add a small offset to the x-coordinate for padding
        ax.text(value, index, f' {value}', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    topic_counts_result = count_topics(csv_file_path)
    if topic_counts_result:
        plot_topics(topic_counts_result)
