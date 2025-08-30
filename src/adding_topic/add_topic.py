#imports

import datasets
import os
from google import genai
import csv
import time
from transformers import AutoTokenizer
import torch # Import torch
from google.genai import types
import pandas as pd # Import pandas
import json # Import json for safer parsing


HF_token = os.environ.get('HF_TOKEN')
G_token = os.environ.get('GOOGLE_API_KEY')

ours = datasets.load_dataset("shachardon/ShareLM")["train"]
print(ours)


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

responses_generated = 1
# Quota limits provided by the user
RPM_LIMIT = 30      # Requests Per Minute
TPM_LIMIT = 15000   # Tokens Per Minute
RPD_LIMIT = 14400   # Requests Per Day

# Variables to track current usage
requests_this_minute = 0
tokens_this_minute = 0
requests_today = 0

# Timestamps to track time for rate limiting
start_time_minute = time.time()
start_time_day = time.time()

MANUAL_START_ROW = 1810
MANUAL_END_ROW = 1816

# File paths for saving progress in Google Drive
progress_csv_path = f"sharelm_dataset_processing_progress.csv"
last_processed_index_path = f"last_processed_index.txt"
original_dataset_csv_path = "sharelm_dataset.csv" # Path to the original saved dataset locally

# Load progress if it exists
start_index = MANUAL_START_ROW # Default start index
if os.path.exists(last_processed_index_path):
    try:
        with open(last_processed_index_path, 'r') as f:
            # Resume from the index AFTER the last successfully processed one
            last_processed = f.read().strip()
            if last_processed:
                start_index = int(last_processed) + 1
        print(f"Resuming processing from index: {start_index}")
    except (ValueError, IOError) as e:
        print(f"Could not load or parse last processed index: {e}. Starting from the beginning of the defined chunk.")
        start_index = MANUAL_START_ROW

# If progress file exists, we will append to it. If not, it will be created.
# We will load the dataset in chunks instead of all at once.

if not os.path.exists(original_dataset_csv_path):
    print("Original dataset CSV not found locally. Loading from Hugging Face and saving to CSV (this might take time).")
    ours = datasets.load_dataset("shachardon/ShareLM")["train"]
    df_ours = ours.to_pandas()
    df_ours.to_csv(original_dataset_csv_path, index=False)
    print(f"Dataset saved to {original_dataset_csv_path}")
    del ours # Free up memory
    del df_ours


client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))


chunk_size = 5
# Write header only if the file is new or we are starting from the very beginning of the manual chunk.
header = not os.path.exists(progress_csv_path) or start_index == MANUAL_START_ROW

model_name = "gemma-3n-e2b-it" # Corrected model name based on traceback
hf_model_name = "google/gemma-3n-e2b-it"
# Initialize a tokenizer using the exact model name and move to the selected device
try:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    # Tokenizer doesn't have a .to(device) method, but the underlying model might.
    # However, for simple tokenization, CPU is usually sufficient and fast.
    print(f"Loaded tokenizer for model: {hf_model_name}.")
except Exception as e:
    print(f"Could not load tokenizer for {hf_model_name}: {e}. Falling back to 'gpt2' tokenizer for demonstration.")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("Loaded 'gpt2' tokenizer as a fallback.")

# Convert 'conversation_metadata' to dictionary if it's a string
def parse_metadata(metadata):
    if isinstance(metadata, str):
        try:
            # Use json.loads for safer parsing of string representation of dictionary
            return json.loads(metadata.replace("'", '"')) # Handle single quotes
        except (json.JSONDecodeError, TypeError):
            # Return an empty dictionary or handle the error as appropriate
            return {}
    elif pd.isna(metadata):
        return {}
    return metadata

# Use a chunksize for reading the CSV to manage memory
# We calculate the number of rows to skip. The +1 is because skiprows is 0-indexed and includes the header.
rows_to_skip = range(1, start_index) if start_index > 0 else None

df_chunk_iter = pd.read_csv(
    progress_csv_path,
    chunksize=chunk_size,
    dtype={'conversation_metadata': 'object'},
    skiprows=rows_to_skip,
    nrows=chunk_size
)

print(f"Starting processing loop from index {start_index} up to {MANUAL_END_ROW}.")
# This flag will be used to break the outer loop
processing_stopped = False
global index
for i, chunk_df in enumerate(df_chunk_iter):
    # Set the DataFrame index to match the absolute index in the original CSV
    # The first chunk's index starts at `start_index`
    chunk_df.index = range(start_index + i * chunk_size, start_index + i * chunk_size + len(chunk_df))

    # conversation_metadata = parse_metadata(chunk_df['conversation_metadata'][i+start_index])



    # Process rows starting from the last processed index within the defined chunk
    for index, row in chunk_df.iterrows():
        conversation_metadata = parse_metadata(chunk_df['conversation_metadata'][index])
        if index >= MANUAL_END_ROW:
            print(f"Reached manual end row {MANUAL_END_ROW}. Stopping.")
            processing_stopped = True
            break # Stop if we've reached the end of our processing window

        # Access the 'conversation' column
        try:
            conversation = row["conversation"]
        except KeyError:
            # Fallback to index if column name is not found
            conversation = row.iloc[1] # Assuming 'conversation' is the second column


        contents = f"""Analyze the following conversation text and classify it as one of the following classes in the comma-separated list [assisting/creative writing, analysis/decision, explanationcoding, factual info, math reason].

        Return ONLY one word referring to the label.

        Conversation: {conversation}
        """
        # Use the tokenizer to get the exact token count
        estimated_tokens_for_prompt = len(tokenizer.encode(contents))
        print(f"Processing row {index}")
        # Check quota before making the API call
        if check_and_update_quota(estimated_tokens_for_prompt):
            max_retries = 5
            response = None # Initialize response to None
            for retry_count in range(max_retries):
                try:
                    response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    )
                    break
                except Exception as e:
                    print(f"Original metadata for row {index}: {conversation_metadata}")
                    print(f"API call failed for row {index} (Attempt {retry_count + 1}/{max_retries}): {e}")
                    if retry_count < max_retries - 1:
                        sleep_time = 2 ** retry_count # Exponential backoff
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        print(f"Max retries reached for row {index}. Skipping.")
                        # print("this is chunck", type(chunk_df.at[index, 'conversation_metadata']))
                        # conversation_metadata = chunk_df.at[index, 'conversation_metadata']
                        conversation_metadata['topic'] = "Error during classification (Failed retries)"
                        chunk_df.at[index, 'conversation_metadata'] = conversation_metadata


            if response is not None and response.text:
                classified_topic = response.text.strip()
                print(f"Processed row {index}: {classified_topic}")
                
                # metadata = chunk_df.at[index, 'conversation_metadata']
                print(f"Original metadata for row {index}: {conversation_metadata}")

                conversation_metadata['topic'] = classified_topic
                chunk_df.at[index, 'conversation_metadata'] = conversation_metadata
                print(f"Updated metadata for row {index}: {conversation_metadata}")

        
        else:
            print(f"Stopping processing at index {index} due to quota limit.")
            processing_stopped = True
            break # Stop processing if quota is exceeded
    
    # --- Save Progress After Each Chunk ---
    # Determine which rows from the chunk were actually processed in this run
    # `index` will hold the last index processed or attempted in the inner loop
    last_processed_index_in_chunk = index
    if processing_stopped and not check_and_update_quota(0): # If stopped by quota, the last item failed
        last_processed_index_in_chunk = index - 1

    # Get the slice of the dataframe that was successfully processed
    processed_chunk_df = chunk_df.loc[chunk_df.index[0]:last_processed_index_in_chunk]

    if not processed_chunk_df.empty:
        # Convert metadata back to string for CSV storage
        processed_chunk_df['conversation_metadata'] = processed_chunk_df['conversation_metadata'].apply(json.dumps)
        processed_chunk_df.to_csv(progress_csv_path, mode='a', header=header, index=False)
        header = False # Header is written, don't write it again
        with open(last_processed_index_path, 'w') as f:
            f.write(str(last_processed_index_in_chunk))
        print(f"Saved processed chunk up to index {last_processed_index_in_chunk}")

    if processing_stopped:
        break # Break the outer loop if processing was stopped

print("\nProcessing finished.")

# The final processed data is in the progress CSV. You can load it to inspect.
if os.path.exists(progress_csv_path):
    print("\nDisplaying first 5 rows from the progress file:")
    df_processed = pd.read_csv(progress_csv_path)
    display(df_processed.head())
else:
    print("No processing was done or progress file was not created.")
    