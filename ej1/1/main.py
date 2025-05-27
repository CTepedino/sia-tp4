import json
import sys
from datetime import datetime
from pathlib import Path

from utils.read_dataset import read_europe_dataset, read_europe_dataset_as_matrix
from kohonen_network import QuadraticKohonenNetwork, HexagonalKohonenNetwork, euclidean_distance, exponential_distance
from utils.normalization import standardize

import numpy as np

networks = {
    "quad": QuadraticKohonenNetwork,
    "hex": HexagonalKohonenNetwork
}

distance_functions = {
    "euclidean": euclidean_distance,
    "exponential": exponential_distance
}

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    seed = config.get("seed", None)
    iterations_per_n = config.get("iterations_per_n", 500)

    countries, data = read_europe_dataset_as_matrix("europe.csv")
    data = standardize(np.array(data))



    network = networks["quad"](
        k=config["k"],
        input_dim=data.shape[1],
        initial_radius=config["initial_r"],
        distance_function=distance_functions[config["distance_fn"]],
        initial_learning_rate=config["initial_lr"]
    )


    network.train(data, iterations=iterations_per_n * data.shape[1], seed=seed)

    clusters = {}

    for i, x in enumerate(data):
        neuron = network.map(x)
        if neuron not in clusters:
            clusters[neuron] = []
        clusters[neuron].append(countries[i])

    for neuron, group in sorted(clusters.items()):
        neuron_int = (int(neuron[0]), int(neuron[1]))
        print(f"Neuron {neuron_int}: {group}")

    results_dir = Path("ej1_1_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = results_dir / f"output_{timestamp}"
    dir_path.mkdir(exist_ok=True)

    config_copy_path = dir_path / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=4)

    clusters_printable = [{
        "neuron_x": int(neuron[0]),
        "neuron_y": int(neuron[1]),
        "countries": countries
        } for neuron, countries in clusters.items()
    ]

    clusters_info_path = dir_path / "clusters.json"
    with open(clusters_info_path, "w") as f:
        json.dump(clusters_printable, f, indent=4)