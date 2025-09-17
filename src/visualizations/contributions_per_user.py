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

    # Create a single figure for the plot
    plt.figure(figsize=(12, 8))

    # Create a horizontal bar chart of the top N user contributions
    user_counts_series.plot(kind='barh')
    plt.title(f'Top {top_n} User Contributions in ShareLM Dataset')
    plt.xlabel('Number of Contributions')
    plt.ylabel('User ID')

    # Invert y-axis to show the user with the most contributions at the top
    plt.gca().invert_yaxis()

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

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
