#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_user_demographics(df, top_n=20):
    """
    Analyzes and plots the location, age, and gender for top N users.

    Args:
        df (pd.DataFrame): The input DataFrame with 'user_id' and 'user_metadata' columns.
        top_n (int): The number of top categories to display for location and age.
    """
    # We only need one entry per user to analyze demographics.
    df_unique_users = df.drop_duplicates(subset=['user_id']).copy()

    # Extract user metadata
    user_meta = df_unique_users['user_metadata'].apply(parse_metadata)
    df_unique_users['location'] = user_meta.apply(lambda x: x.get('location', 'N/A'))
    df_unique_users['age'] = user_meta.apply(lambda x: x.get('age', 'N/A'))
    df_unique_users['gender'] = user_meta.apply(lambda x: x.get('gender', 'N/A'))

    # Filter out rows where location, age, or gender is 'N/A' or there is no content
    df_filtered = df_unique_users[
        (df_unique_users['location'] != 'N/A') & (df_unique_users['age'] != 'N/A') & (df_unique_users['gender'] != 'N/A') &
        (df_unique_users['location'] != '') & (df_unique_users['age'] != '') & (df_unique_users['gender'] != '')
    ].dropna(subset=['user_id'])

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle(f'User Demographics (Top {top_n} Locations & Ages)', fontsize=16)

    # --- 1. Location Plot ---
    # Get the top N locations by user count
    location_counts = df_filtered['location'].value_counts().nlargest(top_n)
    sns.barplot(y=location_counts.index, x=location_counts.values, ax=axes[0], orient='h')
    axes[0].set_title(f'Top {top_n} Location Distribution')
    axes[0].set_xlabel('Number of Users')
    axes[0].set_ylabel('Location')

    # --- 2. Age Plot ---
    # Get the top N age groups by user count
    age_counts = df_filtered['age'].value_counts().nlargest(top_n)
    sns.barplot(y=age_counts.index, x=age_counts.values, ax=axes[1], orient='h')
    axes[1].set_title(f'Top {top_n} Age Distribution')
    axes[1].set_xlabel('Number of Users')
    axes[1].set_ylabel('Age')

    # --- 3. Gender Plot ---
    # Show all gender categories found
    gender_counts = df_filtered['gender'].value_counts()
    sns.barplot(y=gender_counts.index, x=gender_counts.values, ax=axes[2], orient='h')
    axes[2].set_title('Gender Distribution')
    axes[2].set_xlabel('Number of Users')
    axes[2].set_ylabel('Gender')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    try:
        file_path = 'sharelm_dataset_processing_progress.csv'
        print(f"Loading data from '{file_path}'...")
        # Ensure user_metadata is read as a string
        df = pd.read_csv(file_path, dtype={'user_metadata': 'object'}, low_memory=False)
        print("Data loaded successfully.")

        plot_user_demographics(df)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")