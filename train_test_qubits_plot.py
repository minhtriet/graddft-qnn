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
    frozenset(["_180_deg_x_rot"]): to_rgb("red"),
}

NAME_DICT = {
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
    ): "D4",
    frozenset(["_90_deg_x_rot", "_180_deg_x_rot", "_270_deg_x_rot"]): "C3",
    frozenset(["_180_deg_x_rot"]): "180 deg rotation",
}

# Example data as a list of dictionaries
with open("report.json") as f:
    data_list = json.load(f)


def plot_losses(data_entries):
    plt.figure(figsize=(12, 6))
    for data in data_entries:
        # if data[MetricName.N_QUBITS] != 9:
        #     continue
        # Validate Train losses length
        assert (
            len(data[MetricName.TRAIN_LOSSES]) == data[MetricName.EPOCHS]
        ), "Train losses length mismatch"

        # Extract Test losses into x (epochs) and y (losses)
        test_epochs = []
        test_loss_values = []
        for test_dict in data[MetricName.TEST_LOSSES]:
            for epoch, loss in test_dict.items():
                test_epochs.append(int(epoch))
                test_loss_values.append(loss)

        # Get color based on Group members
        group_members = frozenset(data[MetricName.GROUP_MEMBER])
        color = COLOR_DICT[group_members]
        label = NAME_DICT[group_members]

        # Plot training and test losses with the same color
        epochs = range(data[MetricName.EPOCHS])
        plt.plot(
            epochs,
            data[MetricName.TRAIN_LOSSES],
            label=f"Train Loss {label})",
            color=color,
            linewidth=2,
        )
        plt.plot(
            test_epochs,
            test_loss_values,
            label=f"Test Loss {label})",
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

    plt.show()


plot_losses(data_list)
