import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
data = pd.read_csv("smiling2_data.csv", sep='\t')

# Plot the data
plt.plot(data["x"], data["y"])

# Add axis labels and a title
plt.xlabel("Time (ms)")
plt.ylabel("Number of Smiles")
plt.title("Smile Count Over Time")

# Show the plot
plt.show()

# Create pivot table with milliseconds as index and smiles as columns
table = pd.pivot_table(data, values='y', index='x', aggfunc='sum')

# Create heatmap with seaborn
sns.heatmap(table, cmap='YlOrRd', xticklabels=10, yticklabels=10000)
plt.xlabel('Frame')
plt.ylabel('Time (ms)')
plt.title('Smile Count Heatmap')
plt.show()