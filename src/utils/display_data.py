# Specify the path to the CSV file
csv_file_path = "sharelm_dataset_processing_progress.csv"

# Read the CSV file into a pandas DataFrame
df_with_topics = pd.read_csv(csv_file_path, low_memory=False)

pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1500)  # Adjust width as needed
pd.set_option('display.colheader_justify', 'left')  # Optional: Align column headers to the left

# Display the first 5 rows of the DataFrame
print("First 5 rows of the dataset with topics:")
display(df_with_topics.tail(1000)["conversation_metadata"])
os.path.exists("sharelm_dataset_processing_progress.csv")