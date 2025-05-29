import json
import sys
from pathlib import Path

from ej2.letters import pattern_with_noise_percentage
from main import flatten_pattern, unflatten_pattern
from letters import patterns, pattern_with_noise
from hopfield_network import HopfieldNetwork

import numpy as np
import matplotlib.pyplot as plt

def graph_error_by_noise(directory, accuracies_mean, accuracies_deviations, noises, letter):
    plt.figure(figsize=(8, 5))
    plt.errorbar(noises, accuracies_mean, yerr=accuracies_deviations, fmt='-o', capsize=5)
    plt.title(f"Accuracy de {letter}")
    plt.xlabel('Ruido (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(directory / f"error_by_noise_{letter}.png")


def graph_error_by_noise_dual(
        directory,
        noises,
        accuracies_mean_0,
        accuracies_deviations_0,
        label_0,
        accuracies_mean_1,
        accuracies_deviations_1,
        label_1,
        accuracies_mean_2,
        accuracies_deviations_2,
        label_2,
        accuracies_mean_3,
        accuracies_deviations_3,
        label_3,
        accuracies_mean_4,
        accuracies_deviations_4,
        label_4,
        letter
):
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        noises, accuracies_mean_0, yerr=accuracies_deviations_0,
        fmt='-o', capsize=5, label=label_0, color='tab:red'
    )

    # First group
    plt.errorbar(
        noises, accuracies_mean_1, yerr=accuracies_deviations_1,
        fmt='-o', capsize=5, label=label_1, color='tab:blue'
    )

    # Second group
    plt.errorbar(
        noises, accuracies_mean_2, yerr=accuracies_deviations_2,
        fmt='-s', capsize=5, label=label_2, color='tab:orange'
    )

    plt.errorbar(
        noises, accuracies_mean_3, yerr=accuracies_deviations_3,
        fmt='-s', capsize=5, label=label_3, color='#FFFF00'
    )

    plt.errorbar(
        noises, accuracies_mean_4, yerr=accuracies_deviations_4,
        fmt='-s', capsize=5, label=label_4, color='tab:green'
    )

    plt.title(f"Comparación de Accuracy para '{letter}'")
    plt.xlabel('Ruido (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / f"error_by_noise_comparison_{letter}.png")
    plt.close()


def pattern_accuracy(test, goal):
    match = 0
    for a, b in zip(test, goal):
        if a == b:
            match += 1
    return match / len(test)



if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in ["C", "R", "T", "V", "Z"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means0 = []
    deviations0 = []
    noises0= []

    for noise_percent in range(0, 101):
        if noise_percent % 5 == 0:
            noises0.append(noise_percent)
            accuracies = []
            for _ in range(10000):
                test_pattern = flatten_pattern(
                    pattern_with_noise_percentage(config["test_letter"], noise_percent / 100))
                result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
                accuracies.append(pattern_accuracy(result, goal_pattern))

            means0.append(np.mean(accuracies))
            deviations0.append(np.std(accuracies))


    stored_patterns = [flatten_pattern(patterns[letter]) for letter in ["C", "R", "T", "V"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means1 = []
    deviations1 = []
    noises1 = []

    for noise_percent in range(0, 101):
        if noise_percent % 5 == 0:
            noises1.append(noise_percent)
            accuracies = []
            for _ in range(10000):
                test_pattern = flatten_pattern(pattern_with_noise_percentage(config["test_letter"], noise_percent/100))
                result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
                accuracies.append(pattern_accuracy(result, goal_pattern))

            means1.append(np.mean(accuracies))
            deviations1.append(np.std(accuracies))

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in ["C", "T", "V"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means2 = []
    deviations2 = []
    noises2 = []

    for noise_percent in range(0, 101):
        if noise_percent % 5 == 0:
            noises2.append(noise_percent)
            accuracies = []
            for _ in range(10000):
                test_pattern = flatten_pattern(pattern_with_noise_percentage(config["test_letter"], noise_percent/100))
                result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
                accuracies.append(pattern_accuracy(result, goal_pattern))

            means2.append(np.mean(accuracies))
            deviations2.append(np.std(accuracies))

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in ["C", "T", "V"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means3 = []
    deviations3 = []
    noises3 = []

    for noise_percent in range(0, 101):
        if noise_percent % 5 == 0:
            noises2.append(noise_percent)
            accuracies = []
            for _ in range(10000):
                test_pattern = flatten_pattern(
                    pattern_with_noise_percentage(config["test_letter"], noise_percent / 100))
                result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
                accuracies.append(pattern_accuracy(result, goal_pattern))

            means3.append(np.mean(accuracies))
            deviations3.append(np.std(accuracies))

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in ["C", "T", "V"]]

    network = HopfieldNetwork(stored_patterns)

    goal_pattern = flatten_pattern(patterns[config["test_letter"]])

    means4 = []
    deviations4 = []
    noises4 = []

    for noise_percent in range(0, 101):
        if noise_percent % 5 == 0:
            noises2.append(noise_percent)
            accuracies = []
            for _ in range(10000):
                test_pattern = flatten_pattern(
                    pattern_with_noise_percentage(config["test_letter"], noise_percent / 100))
                result = network.get_stored(test_pattern, max_epochs=10, detailed=False)
                accuracies.append(pattern_accuracy(result, goal_pattern))

            means4.append(np.mean(accuracies))
            deviations4.append(np.std(accuracies))

    graph_error_by_noise_dual(
        Path("./"),
        noises1,
        means0,
        deviations0,
        "[C, R, T, V, Z]",
        means1,
        deviations1,
        "[C, R, T, V]",
        means2, deviations2,
        "[C, T, V]",
        means3,
        deviations3,
        "[C, T]",
        means4,
        deviations4,
        "[C]",
        config["test_letter"])
