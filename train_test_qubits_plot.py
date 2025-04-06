import json

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from graddft_qnn.evaluate.metric_name import MetricName

# Define a color dictionary for different group member combinations
COLOR_DICT = {
    frozenset(
        [
            "_90_deg_x_rot",
            "_180_deg_x_rot",
            "_270_deg_x_rot",
            "_180_deg_y_rot",
            "_180_deg_z_rot",
            "y_eq_z_rot",
            "y_eq_neg_z_rot",
        ]
    ): to_rgb("blue"),
    frozenset(["_90_deg_x_rot", "_180_deg_x_rot", "_270_deg_x_rot"]): to_rgb("green"),
}

# Example data as a list of dictionaries
with open("report.json") as f:
    data_list = json.load(f)
    data_list = data_list[39:]


def plot_losses(data_input):
    # Handle both single dict and list of dicts
    data_entries = [data_input] if isinstance(data_input, dict) else data_input

    plt.figure(figsize=(12, 6))

    for data in data_entries:
        # Validate Train losses length
        assert (
            len(data[MetricName.TRAIN_LOSSES]) == data[MetricName.EPOCHS]
        ), "Train losses length mismatch"

        # Extract Test losses into x (epochs) and y (losses)
        test_epochs = []
        test_loss_values = []
        for test_dict in data[MetricName.TEST_LOSSES]:
            for epoch, loss in test_dict.items():
                test_epochs.append(epoch)
                test_loss_values.append(loss)

        # Get color based on Group members
        group_members = frozenset(data[MetricName.GROUP_MEMBER])
        color = COLOR_DICT[group_members]
        label = ", ".join(data[MetricName.GROUP_MEMBER])

        # Plot training and test losses with the same color
        epochs = range(1, data[MetricName.EPOCHS] + 1)
        plt.plot(
            epochs,
            data[MetricName.TRAIN_LOSSES],
            label=f"Train Loss ({label})",
            color=color,
            linewidth=2,
        )
        plt.plot(
            test_epochs,
            test_loss_values,
            label=f"Test Loss ({label})",
            color=color,
            linestyle="--",
            marker="o",
            linewidth=2,
        )

    # Customize the plot
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Losses by Group")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


# Test with a list of dictionaries
plot_losses(data_list)
