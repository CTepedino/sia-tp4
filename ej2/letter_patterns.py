import json
import sys

from ej2.visualize_letter import visualize_letter
from hopfield_network import HopfieldNetwork

def flatten_pattern(matrix):
    return [x for row in matrix for x in row]

def unflatten_pattern(array, rows = 5, cols = 5):
    return [array[i * cols:(i + 1) * cols] for i in range(rows)]

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    patterns = []

    for pattern in config["patterns"]:
        visualize_letter(pattern)
