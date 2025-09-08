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

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Top {top_n} Most Frequent Models in ShareLM Dataset', fontsize=16)

    # --- First subplot: Bar Chart ---
    model_counts_series.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Model Frequency (Bar Chart)')
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Model Name')
    axes[0].invert_yaxis() # Show most frequent at the top

    # --- Second subplot: Scatter Plot ---
    sns.scatterplot(x=model_counts_series.values, y=model_counts_series.index, ax=axes[1])
    axes[1].set_title('Model Distribution (Scatter Plot)')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Model Name')
    axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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