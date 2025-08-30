# counts and analysis

# Create a dictionary to store model counts
model_counts = {}
# Count of rows with valid model names
valid_model_count = 0

# Iterate through the dataset and count model names
for i in range(len(ours)):
    model_name = ours[i]["model_name"]
    if model_name != "":  # Check if model name is not an empty string
        valid_model_count += 1
        if model_name in model_counts:
            model_counts[model_name] += 1
        else:
            model_counts[model_name] = 1

# Print the count of rows with valid model names
print(f"Number of rows with valid model names: {valid_model_count}")

# for j in range(len(ours)):
#     if ours[j]["model_name"] == "":
#         print(ours[j])
#         print(j)
# Sort the model_counts dictionary by value in descending order
sorted_model_counts = dict(sorted(model_counts.items(), key=lambda item: item[1], reverse=True))

# Print the sorted model counts dictionary
print(sorted_model_counts)
print(ours[0])


# check for model name existance

valid_model_rows = []
for i in range(5):
    # if ours[i]["model_name"] != "":
    valid_model_rows.append(ours[i])

print(f"Number of rows with valid model names: {len(valid_model_rows)}")
# Optional: Print the first few rows with valid model names to inspect
if len(valid_model_rows) > 0:
    print("First 5 rows with valid model names:")
    for j in range(min(5, len(valid_model_rows))):
        print(valid_model_rows[j])