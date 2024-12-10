import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv('rewards_before.csv', header=None, names=['episode', 'reward'])

# Define the number of rows and columns for the grid
num_rows = 2
num_cols = 2
total_episodes = len(data)

# We want 4 plots, so divide the data into 4 equal ranges
ranges = [total_episodes // 4 * i for i in range(1, 5)]

# Create subplots (4x4 grid)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot data for each range
for idx, end in enumerate(ranges):
    subset = data[data['episode'] <= end]  # Filter data for the current range
    episodes = subset['episode']
    rewards = subset['reward']
    
    ax = axes[idx]
    ax.plot(episodes, rewards, label=f'Reward per Episode (1-{end})', color='blue', alpha=0.7)

    # Add a trendline
    z = np.polyfit(episodes, rewards, 1)  # Fit a 1st-degree polynomial (linear trendline)
    p = np.poly1d(z)
    ax.plot(episodes, p(episodes), label='Trendline', color='red', linestyle='--')

    # Customize each subplot
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward Trend Over Episodes (1-{end})')
    ax.legend()
    ax.grid(True)

# Hide unused subplots
for idx in range(len(ranges), len(axes)):
    axes[idx].axis('off')

# Adjust layout and display all plots
plt.tight_layout()
plt.show()
