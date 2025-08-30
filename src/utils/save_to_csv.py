# Convert the Hugging Face dataset to a pandas DataFrame
df_ours = ours.to_pandas()

# Save the DataFrame to a CSV file
csv_file_path = "sharelm_dataset.csv"
df_ours.to_csv(csv_file_path, index=False)

print(f"Dataset successfully saved to {csv_file_path}")
