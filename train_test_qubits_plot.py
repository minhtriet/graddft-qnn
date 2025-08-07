import json
from datetime import date, datetime

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
    frozenset(["_180_deg_x_rot_sparse"]): to_rgb("red"),
    frozenset(["_180_deg_x_rot", "_180_deg_y_rot", "_180_deg_z_rot"]): to_rgb("red"),
    frozenset(["naive"]): to_rgb("black"),
}

MARKER_DICT = {6: "D", 9: "X", 3: "1", 12: "o"}

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
    frozenset(["_180_deg_x_rot_sparse"]): "180 deg rotation",
    frozenset(["_180_deg_x_rot", "_180_deg_y_rot", "_180_deg_z_rot"]): "klein",
    frozenset(["naive"]): "Naive",
}

# Example data as a list of dictionaries
with open("report.json") as f:
    data_list = json.load(f)

data_list = [
    data
    for data in data_list
    if datetime.strptime(data["Date"], "%m/%d/%Y, %H:%M:%S").date() > date(2025, 4, 18)
]
data_list = data_list[-1:]


def plot_losses(data_entries):
    plt.figure(figsize=(12, 6))
    for data in data_entries:
        # if data[MetricName.N_QUBITS] not in [6]:
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
        group_members = frozenset(
            data[MetricName.GROUP_MEMBER]
        )  # h4ck! list cannot be key of dict
        color = COLOR_DICT[group_members]
        label = NAME_DICT[group_members]

        # Plot training and test losses with the same color
        epochs = range(data[MetricName.EPOCHS])
        plt.plot(
            epochs,
            data[MetricName.TRAIN_LOSSES],
            label=f"Train Loss {label} {data[MetricName.N_QUBITS]} qb",
            color=color,
            linewidth=2,
            marker=MARKER_DICT[data[MetricName.N_QUBITS]],
        )
        plt.plot(
            test_epochs,
            test_loss_values,
            label=f"Test Loss {label} {data[MetricName.N_QUBITS]} qb",
            color=color,
            linestyle="-.",
            marker=MARKER_DICT[data[MetricName.N_QUBITS]],
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
