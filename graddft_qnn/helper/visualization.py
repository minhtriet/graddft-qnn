import matplotlib.pyplot as plt
import numpy as np


def bar_plot_jvp(jvp_to_plot, filename=None):
    n_bars = len(jvp_to_plot)
    x = np.arange(n_bars)
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x + bar_width, jvp_to_plot.primal.tolist(), bar_width)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
