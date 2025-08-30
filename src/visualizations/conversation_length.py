import matplotlib.pyplot as plt
import seaborn as sns

# Calculate conversation lengths
conversation_lengths = df_ours['conversation'].apply(lambda x: len(x))


# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# First subplot: Horizontal histogram of conversation lengths
n, bins, patches = axes[0].hist(conversation_lengths, bins=[i for i in range(0, 1001, 50)] + [max(conversation_lengths)], orientation='horizontal') # Increased scale for first 1000, then one large bin, added orientation
axes[0].set_title('Distribution of Conversation Lengths in ShareLM Dataset (Histogram)')
axes[0].set_xlabel('Frequency') # Swapped labels
axes[0].set_ylabel('Conversation Length (Number of turns)') # Swapped labels
axes[0].set_ylim(min(bins), max(bins)) # Set y-axis limits to match histogram bins

# Add text labels on each bar in the histogram
for patch in patches:
    x, y = patch.get_xy()
    width = patch.get_width()
    height = patch.get_height()
    if width > 0: # Only label bars with frequency > 0
        axes[0].text(x + width, y + height/2, int(width), va='center', ha='left') # Adjusted text position for horizontal bars

# Second subplot: Horizontal Scatter plot of individual conversation lengths
axes[1].scatter(conversation_lengths, range(len(conversation_lengths)), alpha=0.5) # Swapped x and y for horizontal scatter
axes[1].set_title('Individual Conversation Lengths (Scatter Plot)')
axes[1].set_xlabel('Conversation Length (Number of turns)') # Swapped labels
axes[1].set_ylabel('Conversation Index') # Swapped labels
# Since the scatter plot y-axis represents index, aligning it directly with histogram bins is not straightforward.
# We will match the y-axis range to the total number of conversations for now.
axes[1].set_ylim(0, len(conversation_lengths))
axes[1].set_xlim(axes[0].get_ylim()) # Match x-axis limits with histogram y-axis (conversation length)


plt.tight_layout()
plt.show()



# ==========================================================
# an improved version of the plot
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Filter conversation lengths to include only those between 0 and 1000
short_conversation_lengths = df_ours['conversation'].apply(lambda x: len(x)).loc[lambda x: (x >= 0) & (x <= 1000)].tolist()


# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# First subplot: More detailed horizontal histogram for short conversation lengths
n, bins, patches = axes[0].hist(short_conversation_lengths, bins=100, orientation='horizontal') # Increased number of bins for more detail, added orientation
axes[0].set_title('Distribution of Conversation Lengths (0-1000 turns) in ShareLM Dataset (Histogram)')
axes[0].set_xlabel('Frequency') # Swapped labels
axes[0].set_ylabel('Conversation Length (Number of turns)') # Swapped labels
axes[0].set_ylim(0, 1000) # Set y-axis limits to match the 0-1000 range

# Add text labels on each bar (optional, depending on how crowded it gets)
for patch in patches:
    x, y = patch.get_xy()
    width = patch.get_width()
    height = patch.get_height()
    if width > 0:
        axes[0].text(x + width, y + height/2, int(width), va='center', ha='left', fontsize=8)

# Second subplot: Horizontal Scatter plot of individual short conversation lengths
axes[1].scatter(short_conversation_lengths, range(len(short_conversation_lengths)), alpha=0.5) # Swapped x and y for horizontal scatter
axes[1].set_title('Individual Short Conversation Lengths (0-1000 turns) (Scatter Plot)')
axes[1].set_xlabel('Conversation Length (Number of turns)') # Swapped labels
axes[1].set_ylabel('Conversation Index (within 0-1000 range)') # Swapped labels
# Since the scatter plot y-axis represents index, aligning it directly with histogram bins is not straightforward.
# We will set y-axis limits based on the number of short conversations.
axes[1].set_ylim(0, len(short_conversation_lengths))
axes[1].set_xlim(0, 1000) # Set x-axis limits to match the 0-1000 range of the histogram


plt.tight_layout()
plt.show()
