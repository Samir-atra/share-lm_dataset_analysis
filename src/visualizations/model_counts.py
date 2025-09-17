import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_counts(df, top_n=20):
    """
    Analyzes and plots the distribution of models from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'model_name' column.
        top_n (int): The number of top models to display.
    """
    # Fill any missing model names with 'N/A' for clarity
    df['model_name'] = df['model_name'].fillna('N/A')
    
    # Count model occurrences
    model_counts_series = df['model_name'].value_counts()

    # Sort the series by count for better visualization and select the top N
    model_counts_series = model_counts_series.sort_values(ascending=False).head(top_n)

    # Create a single figure for the plot
    plt.figure(figsize=(12, 8))

    # Create a horizontal bar chart of the top N model counts
    model_counts_series.plot(kind='barh')
    plt.title(f'Top {top_n} Most Frequent Models in ShareLM Dataset')
    plt.xlabel('Count')
    plt.ylabel('Model Name')
    plt.gca().invert_yaxis() # Show most frequent at the top

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    plt.show()

    # Print the name of the most used model separately
    if not model_counts_series.empty:
        most_used_model_name = model_counts_series.index[0]
        print(f"Most used model: {most_used_model_name}")

if __name__ == '__main__':
    try:
        file_path = 'sharelm_with_model.csv'
        print(f"Loading data from '{file_path}'...")
        # Use low_memory=False to avoid mixed type inference issues on large files
        df_ours = pd.read_csv(file_path, low_memory=False)
        print("Data loaded successfully.")

        plot_model_counts(df_ours)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")