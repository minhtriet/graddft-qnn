import os

import matplotlib.pyplot as plt
import numpy as np


def plot_bar_jvp(jvp_to_plot, filename=None):
    n_bars = len(jvp_to_plot)
    x = np.arange(n_bars)
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x + bar_width, jvp_to_plot.primal.tolist(), bar_width)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def plot_list(distances: list, energies: list, fname: str | None = None):
    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        energies,
        marker="o",
        linestyle="-",
        color="blue",
        label="Predicted Energy",
    )
    plt.xlabel("Distance between H atoms (Å)")
    plt.ylabel("Binding Energy (Hartree)")
    plt.title("Binding Energy Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Define the filename
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    if fname:
        base_filename = os.path.join(output_dir, "binding_energy.png")
        plt.savefig(base_filename, dpi=300)
    plt.show()
