"""
Topic vs. Model Visualization Script

This script analyzes the 'sharelm_with_model.csv' dataset to visualize the
relationship between conversation topics and the models used. It generates a
stacked bar chart showing the topic distribution for the top 10 most frequently
used models in the dataset. This helps in understanding which topics are
more common for specific models.

Usage:
    python src/visualizations/topic_per_model.py
"""
import pandas as pd
import json
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

def plot_topics_by_model(df):
    """
    Generates a stacked bar chart showing topic distribution per model.
    """
    # Filter out rows with no topic or 'N/A' model for a cleaner plot
    df_filtered = df[(df['topic'] != 'No Topic Found') & (df['model_name'] != 'N/A')]

    # Get the top 10 models by contribution count
    top_models = df_filtered['model_name'].value_counts().nlargest(10).index
    df_top_models = df_filtered[df_filtered['model_name'].isin(top_models)]

    if df_top_models.empty:
        print("No data available to plot topics by model after filtering.")
        return

    # Group by model and topic to get counts
    model_topic_counts = df_top_models.groupby(['model_name', 'topic']).size().unstack(fill_value=0)

    # Plotting the stacked bar chart
    model_topic_counts.plot(kind='bar', stacked=True, figsize=(14, 8))

    plt.title('Topic Distribution for Top 10 Models')
    plt.xlabel('Model Name')
    plt.ylabel('Number of Conversations')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' was not found.")
    else:
        print(f"Reading data from '{csv_file_path}'...")
        df = pd.read_csv(csv_file_path, dtype={'conversation_metadata': 'object'}, low_memory=False)

        # Create 'topic' and 'model_name' columns for easier processing
        df['topic'] = df['conversation_metadata'].apply(parse_metadata).apply(lambda x: x.get('topic', 'No Topic Found'))
        df['model_name'] = df['model_name'].fillna('N/A')

        plot_topics_by_model(df)
        print("Visualization generated.")
