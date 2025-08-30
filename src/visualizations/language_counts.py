import matplotlib.pyplot as plt
import seaborn as sns

# Count language occurrences
language_counts_series = df_ours['conversation_metadata'].apply(lambda x: x.get('language', 'None')).value_counts()


# Sort the series by count for better visualization and select the top 10
language_counts_series = language_counts_series.sort_values(ascending=False).head(10)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# First subplot: Horizontal bar chart of the top 10 language counts
language_counts_series.plot(kind='barh', ax=axes[0])
axes[0].set_title('Top 10 Most Frequent Languages in ShareLM Dataset (Bar Chart)')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Language')
axes[0].set_ylim(-0.5, len(language_counts_series) - 0.5) # Adjust y-axis limits for bar chart

# Second subplot: Scatter plot of individual language counts
axes[1].scatter(language_counts_series.values, range(len(language_counts_series)))
axes[1].set_title('Individual Language Counts (Scatter Plot)')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Language')
axes[1].set_yticks(range(len(language_counts_series)))
axes[1].set_yticklabels(language_counts_series.index)
axes[1].invert_yaxis() # Invert y-axis to match bar chart order
axes[1].set_ylim(-0.5, len(language_counts_series) - 0.5) # Match y-axis limits with bar chart
axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart


plt.tight_layout()
plt.show()
