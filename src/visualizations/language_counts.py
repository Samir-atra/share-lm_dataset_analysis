import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

def plot_language_distribution(df, top_n=10):
    """
    Analyzes and plots the distribution of languages from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'conversation_metadata' column.
        top_n (int): The number of top languages to display.
    """
    # Count language occurrences by parsing the metadata string
    language_counts_series = df['conversation_metadata'].apply(parse_metadata).apply(lambda x: x.get('language', 'N/A')).value_counts()

    # Sort the series by count for better visualization and select the top N
    language_counts_series = language_counts_series.sort_values(ascending=False).head(top_n)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Top {top_n} Most Frequent Languages in ShareLM Dataset', fontsize=16)

    # First subplot: Horizontal bar chart of the top N language counts
    language_counts_series.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Language Frequency (Bar Chart)')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Language')
    axes[0].invert_yaxis() # Show most frequent at the top

    # Second subplot: Scatter plot of individual language counts
    sns.scatterplot(x=language_counts_series.values, y=language_counts_series.index, ax=axes[1])
    axes[1].set_title('Language Distribution (Scatter Plot)')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Language')
    axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    try:
        file_path = 'sharelm_with_model.csv'
        print(f"Loading data from '{file_path}'...")
        df_ours = pd.read_csv(file_path, dtype={'conversation_metadata': 'object'}, low_memory=False)
        print("Data loaded successfully.")

        plot_language_distribution(df_ours)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
