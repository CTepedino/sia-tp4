import sys
import json
from pathlib import Path
from datetime import datetime

from oja_network import OjaNetwork, decaying_learning_rate, constant_learning_rate
from utils.read_dataset import read_europe_dataset, read_europe_dataset_as_matrix
from utils.normalization import standardize

import numpy as np

learning_rates = {
    "constant": constant_learning_rate,
    "decaying": decaying_learning_rate
}

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    seed = config.get("seed", None)

    countries, data = read_europe_dataset_as_matrix("europe.csv")
    data = standardize(np.array(data))


    print(data)

    network = OjaNetwork(
        input_size=len(data[0]),
        learning_rate_function=learning_rates[config["learning_rate"]],
        seed=seed
    )

    network.train(data, config.get("epochs", 500))

    print(network.weights)

    # results_dir = Path("ej1_2_results")
    # results_dir.mkdir(exist_ok=True)
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # dir_path = results_dir / f"output_{timestamp}"
    # dir_path.mkdir(exist_ok=True)
    #
    # config_copy_path = dir_path / "config.json"
    # with open(config_copy_path, "w") as f:
    #     json.dump(config, f, indent=4)
    #
    # results_info_path = dir_path / "results.json"
    # with open(results_info_path, "w") as f:
    #     json.dump({"weights": [float(w) for w in network.weights]}, f, indent=4)
    #

