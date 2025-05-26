import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def setup_plot(state):
    cmap = ListedColormap(["white", "black"])

    fig, ax = plt.subplots()
    im = ax.imshow(state, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(5) - 0.5, minor=True)
    ax.set_yticks(np.arange(5) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_aspect('equal')


    return fig, ax, im

def visualize_letter(letter, filename = None):
    fig, ax, im = setup_plot(letter)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)


def process_animation(states, filename):
    fig, ax, im = setup_plot(states[0])

    def update(frame):
        im.set_data(states[frame])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(states), interval=1000, blit=True)


    anim.save(filename)

    plt.close(fig)

