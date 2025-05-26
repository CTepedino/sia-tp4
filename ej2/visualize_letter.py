import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def visualize_letter(letter, filename = None):
    cmap = ListedColormap(["white", "black"])

    fig, ax = plt.subplots()
    ax.imshow(letter, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(5) - 0.5, minor=True)
    ax.set_yticks(np.arange(5) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_aspect('equal')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)