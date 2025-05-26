from utils.read_dataset import read_europe_dataset, read_europe_dataset_as_matrix
from kohonen_network import QuadraticKohonenNetwork, HexagonalKohonenNetwork, euclidean_distance, exponential_distance
from utils.normalization import standardize

import numpy as np

networks = {
    "quad": QuadraticKohonenNetwork,
    "hex": HexagonalKohonenNetwork
}

if __name__ == "__main__":

    countries, data = read_europe_dataset_as_matrix("europe.csv")
    data = standardize(np.array(data))

    network = networks["quad"](
        k=4,
        input_dim=data.shape[1],
        initial_radius=4,
        distance_function=euclidean_distance,
        initial_learning_rate=0.5
    )


    network.train(data, iterations=1000, seed=42)

    clusters = {}

    for i, x in enumerate(data):
        neuron = network.map(x)
        if neuron not in clusters:
            clusters[neuron] = []
        clusters[neuron].append(countries[i])

    for neuron, group in sorted(clusters.items()):
        neuron_int = (int(neuron[0]), int(neuron[1]))
        print(f"Neuron {neuron_int}: {group}")

