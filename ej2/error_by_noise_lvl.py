import json
import sys
from pathlib import Path

from main import flatten_pattern, unflatten_pattern
from letters import patterns, pattern_with_noise
from hopfield_network import HopfieldNetwork

import numpy as np
import matplotlib.pyplot as plt

def graph_error_by_noise(directory, accuracies_mean, accuracies_deviations, noises, letter):
    plt.figure(figsize=(8, 5))
    plt.errorbar(noises, accuracies_mean, yerr=accuracies_deviations, fmt='-o', capsize=5)
    plt.title(f"Accuracy de {letter}")
    plt.xlabel('Ruido (pixeles)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(directory / f"error_by_noise_{letter}.png")

def pattern_accuracy(test, goal):
    match = 0
    for a, b in zip(test, goal):
        if a == b:
            match += 1
    return match / len(test)



if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in config["stored_letters"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means = []
    deviations = []
    noises = []

    for noise_level in range(11):
        noises.append(noise_level)
        accuracies = []
        for _ in range(10000):
            test_pattern = flatten_pattern(pattern_with_noise(config["test_letter"], noise_level))
            result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
            accuracies.append(pattern_accuracy(result, goal_pattern))

        means.append(np.mean(accuracies))
        deviations.append(np.std(accuracies))

    graph_error_by_noise(Path("./"), means, deviations, noises, config["test_letter"])
