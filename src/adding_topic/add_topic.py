#Imports

import datasets
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from google import genai
import csv
import time
from transformers import AutoTokenizer
import torch
from google.genai import types
import pandas as pd 
import json
from src.utils.quota_update import check_and_update_quota


HF_token = os.environ.get('HF_TOKEN')
G_token = os.environ.get('GOOGLE_API_KEY')

ours = datasets.load_dataset("shachardon/ShareLM")["train"]
print(ours)


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MANUAL_START_ROW = 0
MANUAL_END_ROW = 10161

# --- Execution Mode ---
# The script operates in one of two modes, controlled by the 'runs' variable.
#
# "first_run": 
#   - This is the primary mode for the initial processing of the dataset.
#   - It iterates through the dataset sequentially, starting from where it last left off.
#   - It calls the API to classify each conversation and adds a 'topic'.
#   - If an API call fails or a topic isn't found, it marks the row accordingly.
# "second_run":
#   - This is a cleanup and retry mode.
#   - It specifically targets rows that were marked with "Error: API call failed" or "No Topic Found" during a previous run.
#   - It attempts to re-process only these failed rows to complete the dataset.
runs = "second_run"
# File paths for saving progress in Google Drive
progress_csv_path = f"sharelm_with_model.csv"
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

if runs == "first_run":
    # Use a chunksize for reading the CSV to manage memory
    # We calculate the number of rows to skip. The +1 is because skiprows is 0-indexed and includes the header.
    rows_to_skip = range(1, start_index) if start_index > 0 else None

    df_chunk_iter = pd.read_csv(
        original_dataset_csv_path, 
        chunksize=chunk_size,
        skiprows=rows_to_skip,
        nrows=MANUAL_END_ROW - start_index,
        dtype={'conversation_metadata': 'object'},
    )

    print(f"Starting processing loop from index {start_index} up to {MANUAL_END_ROW}.")
    # This flag will be used to break the outer loop
    processing_stopped = False
    for i, chunk_df in enumerate(df_chunk_iter):
        # Set the DataFrame index to match the absolute index in the original CSV
        # The first chunk's index starts at `start_index`

        chunk_df.index = range(start_index + i * chunk_size, start_index + i * chunk_size + len(chunk_df))
        
        # Process rows starting from the last processed index within the defined chunk
        for index, row in chunk_df.iterrows():
            conversation_metadata = parse_metadata(row['conversation_metadata'])
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
                            conversation_metadata['topic'] = "Error: API call failed"
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
        last_processed_index_in_chunk = chunk_df.index[-1]
        if processing_stopped: # If we stopped early (e.g., quota, manual end)
            # The last successfully processed index is the one before the loop broke
            last_processed_index_in_chunk = index - 1 if 'index' in locals() and index > chunk_df.index[0] else chunk_df.index[0] -1

        # Get the slice of the dataframe that was successfully processed
        processed_chunk_df = chunk_df.loc[chunk_df.index[0]:last_processed_index_in_chunk] if last_processed_index_in_chunk >= chunk_df.index[0] else pd.DataFrame()

        if not processed_chunk_df.empty:
            # --- Read-Modify-Write to update the main progress file ---
            
            # First, prepare the processed chunk by converting metadata to JSON strings
            processed_chunk_df_copy = processed_chunk_df.copy()
            processed_chunk_df_copy['conversation_metadata'] = processed_chunk_df_copy['conversation_metadata'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

            if os.path.exists(progress_csv_path):
                # Load the entire existing progress file to update it.
                # This is a trade-off for correctness and simplicity over stream-processing the write.
                # Given the chunk-based processing, this memory hit happens only during the save step.
                main_df = pd.read_csv(progress_csv_path, low_memory=False)
                # Use update() to modify the rows in place based on the index.
                main_df.update(processed_chunk_df_copy)
                # Save the entire updated DataFrame, overwriting the old file.
                main_df.to_csv(progress_csv_path, index=False, quoting=csv.QUOTE_ALL)
            else:
                # If progress file doesn't exist, create it from the original and update it.
                print(f"Progress file not found. Creating a new one from the original dataset.")
                main_df = pd.read_csv(original_dataset_csv_path, low_memory=False)
                main_df.update(processed_chunk_df_copy)
                main_df.to_csv(progress_csv_path, index=False, quoting=csv.QUOTE_ALL)

            with open(last_processed_index_path, 'w') as f:
                f.write(str(last_processed_index_in_chunk))
            print(f"Saved processed chunk up to index {last_processed_index_in_chunk}")

        if processing_stopped:
            break # Break the outer loop if processing was stopped

elif runs == "second_run":
    # This block handles the second pass over the data, focusing only on rows that need reprocessing.
    print("Starting second run to process failed or missing topics...")
    if not os.path.exists(progress_csv_path):
        print(f"Error: Progress file '{progress_csv_path}' not found. Please run with 'first_run' first.")
    else:
        # Load the entire progress CSV into a pandas DataFrame.
        # This is necessary because the rows to be processed are scattered throughout the file, not sequential.
        df = pd.read_csv(progress_csv_path, low_memory=False, dtype={'conversation_metadata': 'object'})
        
        # Define the specific topic values that indicate a row needs to be re-processed.
        topics_to_reprocess = ["Error: API call failed", "No Topic Found"]
        
        # Helper function to check the 'topic' within the metadata JSON string.
        def check_topic(metadata_str):
            metadata = parse_metadata(metadata_str)
            topic = metadata.get('topic')
            # Return True if the topic is one of the ones we want to reprocess, or if no topic exists at all.
            return topic in topics_to_reprocess or topic is None

        # Apply the check function to the 'conversation_metadata' column to create a boolean mask.
        rows_to_reprocess_mask = df['conversation_metadata'].apply(check_topic)
        # Use the mask to filter the DataFrame, creating a new view containing only the rows to be re-processed.
        df_to_reprocess = df[rows_to_reprocess_mask]

        print(f"Found {len(df_to_reprocess)} rows to re-process.")

        if not df_to_reprocess.empty:
            # Iterate over the filtered DataFrame of rows that need reprocessing.
            for index, row in df_to_reprocess.iterrows():
                conversation_metadata = parse_metadata(row['conversation_metadata'])
                conversation = row["conversation"]

                contents = f"""Analyze the following conversation text and classify it as one of the following classes in the comma-separated list [assisting/creative writing, analysis/decision, explanationcoding, factual info, math reason].

                Return ONLY one word referring to the label.

                Conversation: {conversation}
                """

                estimated_tokens_for_prompt = len(tokenizer.encode(contents))
                print(f"Re-processing row {index}")

                # Check API quota before making a call.
                if check_and_update_quota(estimated_tokens_for_prompt):
                    max_retries = 5
                    response = None
                    # The API call and retry logic is the same as in the first run.
                    for retry_count in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=contents,
                            )
                            break
                        except Exception as e:
                            print(f"API call failed again for row {index} (Attempt {retry_count + 1}/{max_retries}): {e}")
                            if retry_count < max_retries - 1:
                                time.sleep(2 ** retry_count)
                            else:
                                # If all retries fail, update the topic to a new error state.
                                conversation_metadata['topic'] = "Error: API call failed again"
                                # Update the main DataFrame directly at the specific index. The metadata is converted back to a JSON string.
                                df.at[index, 'conversation_metadata'] = json.dumps(conversation_metadata)

                    # If the API call was successful, update the topic in the main DataFrame.
                    if response is not None and response.text:
                        classified_topic = response.text.strip()
                        print(f"Successfully re-processed row {index}: {classified_topic}")
                        conversation_metadata['topic'] = classified_topic
                        # Update the 'conversation_metadata' cell for the specific row in the main DataFrame.
                        df.at[index, 'conversation_metadata'] = json.dumps(conversation_metadata)
                else:
                    # If quota is hit, stop processing and break the loop. The changes made so far will still be saved.
                    print(f"Stopping re-processing at index {index} due to quota limit.")
                    break

            # After the loop finishes (or is broken), save the entire modified DataFrame back to the CSV.
            print("Saving updated data after second run...")
            df.to_csv(progress_csv_path, index=False, quoting=csv.QUOTE_ALL)
            print("Save complete.")

print("\nProcessing finished.")

# You can now use the display_data.py script to view the contents of sharelm_dataset_processing_progress.csv
print(f"To view the updated data, run: python src/utils/display_data.py")
