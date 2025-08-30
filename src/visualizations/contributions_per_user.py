import matplotlib.pyplot as plt
import seaborn as sns

# Count user contributions
user_counts_series = df_ours['user_id'].value_counts()

# Select the top 20 user contributions
user_counts_series = user_counts_series.sort_values(ascending=False).head(20)

# Create a figure with two subplots arranged in a single row
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# First subplot: Horizontal bar chart of the top 20 user contributions
user_counts_series.plot(kind='barh', ax=axes[0])
axes[0].set_title('Top 20 Most Frequent User IDs in ShareLM Dataset (Bar Chart)')
axes[0].set_xlabel('Number of Contributions')
axes[0].set_ylabel('User ID')
axes[0].set_ylim(-0.5, len(user_counts_series) - 0.5) # Adjust y-axis limits for bar chart

# Second subplot: Scatter plot of individual user contribution counts
axes[1].scatter(user_counts_series.values, range(len(user_counts_series)))
axes[1].set_title('Individual User Contribution Counts (Scatter Plot)')
axes[1].set_xlabel('Number of Contributions')
axes[1].set_ylabel('User ID')

# Set y-axis tick locations and labels for the scatter plot to display User IDs
axes[1].set_yticks(range(len(user_counts_series)))
axes[1].set_yticklabels(user_counts_series.index)

# Invert the y-axis of the scatter plot to match the order of the horizontal bar chart
axes[1].invert_yaxis()
axes[1].set_ylim(-0.5, len(user_counts_series) - 0.5) # Match y-axis limits with bar chart
axes[1].set_xlim(axes[0].get_xlim()) # Match x-axis limits with bar chart


# Adjust subplot parameters for a tight layout
plt.tight_layout()

# Display the figure
plt.show()
