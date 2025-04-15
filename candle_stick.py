import json

import matplotlib.pyplot as plt

with open("report.json") as f:
    data = json.load(f)


# Function to extract test losses from a group
def extract_test_losses(group):
    # Extract the list of dictionaries under 'Test losses'
    test_losses_dicts = group["Test losses"]
    # Get the loss values (values from each dictionary)
    losses = [list(d.values())[0] for d in test_losses_dicts]
    return losses


# Process the data
all_losses = []
labels = []

NUM_MEM_GROUP_MAP = {3: "C3", 7: "D4"}

for i, group in enumerate(data, 1):  # Start enumeration at 1 for group numbering
    # Extract test losses
    losses = extract_test_losses(group)
    all_losses.append(losses)

    # Create a label with qubit number and group member count
    num_qubits = group["Num qubits"]
    num_members = len(group["Group members"])
    label = f"({num_qubits} qubits, group {NUM_MEM_GROUP_MAP[num_members]})"
    labels.append(label)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot boxplot
ax.boxplot(all_losses, vert=True, patch_artist=True, labels=labels)

# Customize the plot
ax.set_ylabel("Test Loss")
ax.set_title("Boxplot of Test Losses by Group")
ax.grid(True, linestyle="--", alpha=0.7)

# Rotate x-axis labels if there are many groups
plt.xticks(rotation=15, ha="right")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
