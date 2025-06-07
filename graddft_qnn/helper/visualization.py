import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np

from datasets import DatasetDict
from graddft_qnn.cube_dataset.h2_multibond import H2MultibondDataset

DISTANCES = np.arange(0.1, 4.0, 0.1)
# this file can be created by running Classical_No_Down.py
CLASSICAL_WITH_DOWN_FILENAME = "classical_with_down.json"


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

    # Plot each set of energies
    for i, energy_dict in enumerate(energies):
        plt.plot(
            [float(x) for x in energy_dict.keys()],
            energy_dict.values(),
            color=colors[i],  # Cycle through colors if needed
            label=labels[i],  # Use corresponding label for the legend
        )

    plt.xlabel("Distance between H atoms (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title("Potential Energy Curves")
    plt.grid(alpha=0.3)
    plt.legend()  # Display legend with all labels
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(tck.MultipleLocator(1))
    ax.xaxis.set_major_formatter(tck.FuncFormatter(lambda x, pos: f"{x:.2f}"))

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
            if dist in [0.944865, 1.0960434, 1.0204542]:
                continue
            h2_dist_energy[dist] = data["groundtruth"]

    return dict(sorted(h2_dist_energy.items()))
