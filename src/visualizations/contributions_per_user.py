import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_user_contributions(df, top_n=20):
    """
    Analyzes and plots the top N user contributions from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'user_id' column.
        top_n (int): The number of top contributors to display.
    """
    # Count user contributions
    user_counts_series = df['user_id'].value_counts()

    # Select the top N user contributions
    user_counts_series = user_counts_series.sort_values(ascending=False).head(top_n)

    # Create a figure with two subplots arranged in a single row
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Top {top_n} User Contributions in ShareLM Dataset', fontsize=16)

    # First subplot: Horizontal bar chart of the top N user contributions
    user_counts_series.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Contribution Frequency (Bar Chart)')
    axes[0].set_xlabel('Number of Contributions')
    axes[0].set_ylabel('User ID')

    # Second subplot: Scatter plot of individual user contribution counts
    sns.scatterplot(x=user_counts_series.values, y=user_counts_series.index, ax=axes[1])
    axes[1].set_title('Contribution Distribution (Scatter Plot)')
    axes[1].set_xlabel('Number of Contributions')
    axes[1].set_ylabel('User ID')

    # Match axes for consistency (most contributions at the top)
    axes[0].invert_yaxis()
    axes[1].set_xlim(axes[0].get_xlim())

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the figure
    plt.show()

if __name__ == '__main__':
    # --- IMPORTANT ---
    # You need to load your data here.
    # Replace 'path/to/your/data.parquet' with the actual path to your data file.
    # If your data is in a different format (e.g., CSV), use pd.read_csv().
    file_path = 'sharelm_with_model.csv' 
    df_ours = pd.read_csv(file_path)

    plot_user_contributions(df_ours)
