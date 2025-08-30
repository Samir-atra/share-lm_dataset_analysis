from google.colab import files

# Specify the path to the CSV file you saved
csv_file_path = "sharelm_dataset.csv"

try:
  files.download(csv_file_path)
  print(f"Initiated download for {csv_file_path}. Check your browser's downloads.")
except Exception as e:
  print(f"An error occurred during download: {e}")