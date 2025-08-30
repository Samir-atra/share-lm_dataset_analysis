import matplotlib.pyplot as plt
import seaborn as sns

# Convert the dictionary to a pandas Series for easier plotting
model_counts_series = df_ours['model_name'].value_counts()

# Sort the series by count for better visualization and select the top 20
model_counts_series = model_counts_series.sort_values(ascending=False).head(20)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# First subplot: Horizontal bar chart of the top 20 model counts
model_counts_series.plot(kind='barh', ax=axes[0])
axes[0].set_title('Top 20 Most Frequent Models in ShareLM Dataset (Bar Chart)')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Model Name')
axes[0].set_ylim(-0.5, len(model_counts_series) - 0.5) # Adjust y-axis limits for bar chart

# Add the most used model name as text annotation in the bar chart
most_used_model_name = model_counts_series.index[0]
most_used_model_count = model_counts_series.iloc[0]
axes[0].text(most_used_model_count + 1, 0, f'{most_used_model_name} ({most_used_model_count})', va='center', ha='left', fontsize=10, color='black')

# Second subplot: Scatter plot of individual model counts
axes[1].scatter(model_counts_series.values, range(len(model_counts_series)))
axes[1].set_title('Individual Model Counts (Scatter Plot)')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Model Name')
axes[1].set_yticks(range(len(model_counts_series)))
axes[1].set_yticklabels(model_counts_series.index)
axes[1].invert_yaxis() # Invert y-axis to match bar chart order
axes[1].set_ylim(-0.5, len(model_counts_series) - 0.5) # Match y-axis limits with bar chart
axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart


plt.tight_layout()
plt.show()

# Print the name of the most used model separately
print(f"Most used model: {most_used_model_name}")


# =========================================================================================
# improved version of the model count visualization
# =========================================================================================


import matplotlib.pyplot as plt
import seaborn as sns

# Filter out the most used model and models with empty names
filtered_model_counts_series = model_counts_series[1:] # Exclude the first (most used) model
filtered_model_counts_series = filtered_model_counts_series[filtered_model_counts_series.index != ''] # Exclude empty names

# Select the top 20 from the filtered list
filtered_model_counts_series = filtered_model_counts_series.head(20)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# First subplot: Horizontal bar chart of the filtered top 20 model counts
filtered_model_counts_series.plot(kind='barh', ax=axes[0])
axes[0].set_title('Top 20 Most Frequent Models (Excluding Most Used and Empty Names) (Bar Chart)')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Model Name')
axes[0].set_ylim(-0.5, len(filtered_model_counts_series) - 0.5) # Adjust y-axis limits for bar chart

# Second subplot: Scatter plot of individual filtered model counts
axes[1].scatter(filtered_model_counts_series.values, range(len(filtered_model_counts_series)))
axes[1].set_title('Individual Filtered Model Counts (Scatter Plot)')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Model Name')
axes[1].set_yticks(range(len(filtered_model_counts_series)))
axes[1].set_yticklabels(filtered_model_counts_series.index)
axes[1].invert_yaxis() # Invert y-axis to match bar chart order
axes[1].set_ylim(-0.5, len(filtered_model_counts_series) - 0.5) # Match y-axis limits with bar chart
axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart


plt.tight_layout()
plt.show()