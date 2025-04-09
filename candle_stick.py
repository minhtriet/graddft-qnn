import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

dates = pd.date_range('2025-04-01', periods=10, freq='D')  # 10 dates
open_prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
high_prices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
low_prices = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
close_prices = [10, 12, 12, 14, 16, 16, 18, 19, 19, 18]



for data in data_entries:
    if data[MetricName.N_QUBITS] != 9:
        continue
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

# Stack the data for boxplot
candlestick_data = []
for i in range(len(dates)):
    # Each element is [low, open, close, high]
    candlestick_data.append([low_prices[i], open_prices[i], close_prices[i], high_prices[i]])

# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(candlestick_data, positions=np.arange(len(dates)), widths=0.5, patch_artist=True,
           medianprops=dict(color="black", linewidth=2), boxprops=dict(facecolor='green', color='green'),
           whiskerprops=dict(color='black', linewidth=1))

# Set x-axis to be the dates
ax.set_xticks(np.arange(len(dates)))
ax.set_xticklabels(dates.strftime('%Y-%m-%d'), rotation=45)

# Set title and labels
ax.set_title('Candlestick Chart Using Boxplot')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

# Show the plot
plt.tight_layout()
plt.show()
