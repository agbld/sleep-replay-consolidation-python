#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import Greens
from matplotlib import font_manager as fm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Render results')
parser.add_argument('--results', type=str, default='./results.csv', help='Path to the results CSV file')
parser.add_argument('--output', type=str, default='./results.png', help='Path to the output PNG file')
args = parser.parse_args()

# Load the CSV file into a DataFrame, selecting specific columns
df = pd.read_csv(args.results, usecols=['Approach', 'Stage', 'steps', 'Task 0', 'Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task All'])
df = df[['Approach', 'Stage', 'steps', 'Task 0', 'Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task All']]
df = df[~df['Stage'].str.contains('Initial')]
df = df[~df['Stage'].str.contains('Synthetic')]
df = df.reset_index(drop=True)


# Set the style of the plot
sns.set_theme(style="white")

# Create a figure with a specific size
fig, ax = plt.subplots(figsize=(12, 10))  # Adjust size to fit your table

# Hide the axes
ax.axis('tight')
ax.axis('off')

# Normalize and create a colormap for cell coloring
norm = Normalize(vmin=0, vmax=100)
color_map = Greens

# Create the table, aligning text to center
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)  # Adjust font size as needed
table.scale(1, 2)  # Adjust scaling to fit your text
for key, cell in table.get_celld().items():
    if key[1] in [0]:  # Indices of Task 0 to Task 4 columns
        cell.set_width(0.35)  # Set a different width for these specific columns
    if key[1] in [1]:
        cell.set_width(0.25)

# Color formatting for task columns based on value
task_cols = [df.columns.get_loc(col) for col in df.columns if 'Task' in col or col == 'All Task']
for (i, j), val in np.ndenumerate(df.values):
    if j in task_cols and np.isreal(val):
        color = color_map(norm(val) - 0.2)
        table[(i+1, j)].set_facecolor(color)
        table[(i+1, j)].set_text_props(color='black') # if val > 50 else 'white')

# Make all font bold in rows with "After Training" under "Stage"
after_training_rows = df.index[df['Stage'].str.contains('After Training') |
                               df['Stage'].str.contains('Merge') |
                               df['Stage'].str.contains('Concat')].tolist()
for row in after_training_rows:
    for col in range(len(df.columns)):
        table[(row + 1, col)].set_text_props(weight='bold')
for col in range(len(df.columns)):
    table[(0, col)].set_text_props(weight='bold')

# Save the figure
plt.savefig(args.output, bbox_inches='tight', dpi=300)

# %%
