import copy
import sys
import json
from pathlib import Path
from datetime import datetime

from oja_network import OjaNetwork, decaying_learning_rate, constant_learning_rate
from utils.read_dataset import read_europe_dataset, read_europe_dataset_as_matrix
from utils.normalization import standardize
from pc1_graphics import generate_pc1_graphics

import numpy as np
from sklearn.decomposition import PCA

learning_rates = {
    "constant": constant_learning_rate,
    "decaying": decaying_learning_rate
}

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    seed = config.get("seed", None)

    countries, data = read_europe_dataset_as_matrix("europe.csv")
    data_scaled = standardize(np.array(data))

    pca = PCA()
    pca.fit(data_scaled)
    library_pc1 = pca.components_[0]
    library_pc1_scores = pca.transform(data_scaled)[:, 0]

    network = OjaNetwork(
        input_size=len(data[0]),
        initial_lr=config["initial_lr"],
        learning_rate_function=learning_rates[config["learning_rate"]],
        seed=seed
    )
    network.train(data_scaled, config.get("epochs", 500))
    oja_pc1 = network.weights
    oja_pc1_scores = data_scaled @ oja_pc1


    similarity = np.dot(
        oja_pc1 / np.linalg.norm(oja_pc1),
        library_pc1 / np.linalg.norm(library_pc1)
    )

    results_dir = Path("ej1_2_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = results_dir / f"output_{timestamp}"
    dir_path.mkdir(exist_ok=True)

    config_copy_path = dir_path / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=4)


    results_info_path = dir_path / "results.json"
    with open(results_info_path, "w") as f:
        json.dump({
            "oja_pc1": [float(w) for w in oja_pc1],
            "library_pc1": [float(w) for w in library_pc1],
            "cosine_similarity": similarity
        }, f, indent=4)

    library_graphs_path = dir_path / "library"
    library_graphs_path.mkdir(exist_ok=True)
    generate_pc1_graphics(library_graphs_path, library_pc1, library_pc1_scores, countries)

    oja_graphs_path = dir_path / "oja"
    oja_graphs_path.mkdir(exist_ok=True)
    generate_pc1_graphics(oja_graphs_path, oja_pc1, oja_pc1_scores, countries)

