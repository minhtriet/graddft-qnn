import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from datasets import DatasetDict
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset

DISTANCES = np.arange(0.5, 4.0, 0.1)
# this file can be created by running Classical_No_Down.py
CLASSICAL_WITH_DOWN_FILENAME = "classical_with_down.json"
classical = [
    0.9658405,
    0.19309501,
    -0.14557038,
    -0.31851137,
    -0.41391742,
    -0.468589,
    -0.5002356,
    -0.5181826,
    -0.52766836,
    -0.5318055,
    -0.532502,
    -0.5309742,
    -0.5280177,
    -0.5241591,
    -0.51974666,
    -0.51500493,
    -0.5100828,
    -0.50508165,
    -0.5001214,
    -0.49517483,
    -0.49033344,
    -0.4856318,
    -0.481096,
    -0.4767468,
    -0.47259367,
    -0.46864387,
    -0.46489888,
    -0.4613584,
    -0.45802045,
    -0.4548809,
    -0.45193544,
    -0.44917935,
    -0.4466065,
    -0.44422832,
    -0.44201794,
    -0.43995872,
    -0.43805748,
    -0.43630767,
    -0.43470174,
    -0.43313012,
    -0.43179032,
    -0.4305719,
    -0.42958814,
    -0.42859116,
    -0.4276937,
    -0.42688844,
    -0.42616805,
    -0.42530146,
]


def plot_bar_jvp(jvp_to_plot, filename=None):
    n_bars = len(jvp_to_plot)
    x = np.arange(n_bars)
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x + bar_width, jvp_to_plot.primal.tolist(), bar_width)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_list(
    energies: list[dict],
    labels: list[str] | None = None,
    fname: str = "plot.png",
):
    plt.figure(figsize=(10, 6))

    colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta", "brown"]

    assert len(energies) == len(
        labels
    ), "The number of energy lists must match the number of labels."

    # Plot each set of energies
    for i, energy_dict in enumerate(energies):
        plt.plot(
            energy_dict.keys(),
            energy_dict.values(),
            marker="o",
            linestyle="-",
            color=colors[i],  # Cycle through colors if needed
            label=labels[i],  # Use corresponding label for the legend
        )

    plt.xlabel("Distance between H atoms (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Potential Energy Curves")
    plt.grid(alpha=0.3)
    plt.legend()  # Display legend with all labels
    plt.tight_layout()

    # Define the filename and save if provided
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    if fname:
        base_filename = os.path.join(
            output_dir, fname if fname.endswith(".png") else f"{fname}.png"
        )
        plt.savefig(base_filename, dpi=300)

    plt.show()


def h2_dist_energy():
    if pathlib.Path("datasets/h2_dataset").exists():
        dataset = DatasetDict.load_from_disk(pathlib.Path("datasets/h2_dataset"))
    else:
        dataset = H2MultibondDataset.get_dataset()
        dataset.save_to_disk("datasets/h2_dataset")

    h2_dist_energy = {}
    for key in ["train", "test"]:
        for data in dataset[key]:
            dist = np.linalg.norm(
                np.array(data["coordinates"][0]) - np.array(data["coordinates"][1])
            )
            h2_dist_energy[dist] = data["groundtruth"]

    return dict(sorted(h2_dist_energy.items()))
