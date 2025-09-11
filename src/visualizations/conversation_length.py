import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_conversation_lengths(df, max_len=100, hist_bins=100):
    """
    Analyzes and plots the distribution of conversation lengths from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'conversation' column.
        max_len (int): The maximum conversation length to include in the plot.
        hist_bins (int): The number of bins for the histogram.
    """
    # The 'conversation' column might be a string representation of a list.
    # We need to safely evaluate it.
    try:
        # Attempt to evaluate the string as a Python literal (list of dicts)
        conversation_lengths = df['conversation'].apply(lambda x: len(json.loads(x.replace("'", '"'))))
    except (json.JSONDecodeError, AttributeError):
        # Fallback if it's already a list or another error occurs
        print("Could not parse 'conversation' as JSON string, attempting direct length calculation.")
        conversation_lengths = df['conversation'].apply(len)
        
        turns = []
        index = []
        s = df["conversation"]
        for i in range(9977):
            turns.append(s[i].count("content"))
        indecies = enumerate(turns)
        for i in list(indecies):
            index.append(i[0])
        conversation_lengths = pd.Series(turns, index=index)


    # Filter for conversations up to max_len for better visualization
    short_conversation_lengths = conversation_lengths[conversation_lengths <= max_len]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Analysis of Conversation Lengths (Up to {max_len} turns)', fontsize=16)

    # --- First subplot: Histogram ---
    sns.histplot(data=short_conversation_lengths, bins=hist_bins, ax=axes[0])
    axes[0].set_title(f'Distribution of Conversation Lengths (0-{max_len} turns)')
    axes[0].set_xlabel('Conversation Length (Number of turns)')
    axes[0].set_ylabel('Frequency')

    # --- Second subplot: Scatter Plot ---
    # Create a DataFrame for easier plotting with seaborn
    scatter_df = pd.DataFrame({
        'length': short_conversation_lengths,
        'index': range(len(short_conversation_lengths))
    })
    sns.scatterplot(x='index', y='length', data=scatter_df, alpha=0.5, ax=axes[1], s=10)
    axes[1].set_title(f'Individual Conversation Lengths (0-{max_len} turns)')
    axes[1].set_xlabel('Conversation Index')
    axes[1].set_ylabel('Conversation Length (Number of turns)')
    axes[1].set_ylim(0, max_len)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    try:
        file_path = 'sharelm_with_model.csv'
        print(f"Loading data from '{file_path}'...")
        df_ours = pd.read_csv(file_path)
        print("Data loaded successfully.")

        plot_conversation_lengths(df_ours)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
