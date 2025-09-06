import pandas as pd
import os

# --- Manual Selection ---
# Set the start and end row numbers you want to display.
# Note: These are based on the row index, starting from 0.
START_ROW = 1809
END_ROW = 1820 # This row will be included in the output

# Specify the path to the CSV file
csv_file_path = "sharelm_dataset_processing_progress.csv"
# csv_file_path = "sharelm_dataset.csv"

if os.path.exists(csv_file_path):
    # Read the CSV file into a pandas DataFrame
    df_with_topics = pd.read_csv(csv_file_path, low_memory=False)

    # Set pandas display options to show full content
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1500)
    pd.set_option('display.colheader_justify', 'left')

    # Display the selected slice of the DataFrame
    print(f"Displaying rows from {START_ROW} to {END_ROW} of the dataset with topics:")
    print(df_with_topics.iloc[START_ROW:END_ROW+1])
else:
    print(f"Error: The file '{csv_file_path}' was not found.")
