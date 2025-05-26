import json
import sys
from datetime import datetime
from pathlib import Path

from ej2.hopfield_network import HopfieldNetwork
from ej2.letters import patterns, pattern_with_noise
from ej2.visualize_letter import visualize_letter, process_animation
from utils.json_matrix_encoder import CompactRowsEncoder


def flatten_pattern(matrix):
    return [x for row in matrix for x in row]

def unflatten_pattern(array, rows = 5, cols = 5):
    return [array[i * cols:(i + 1) * cols] for i in range(rows)]

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    seed = config.get("seed", None)

    stored_patterns = [flatten_pattern(patterns[letter]) for letter in config["stored_letters"]]

    test_pattern = flatten_pattern(pattern_with_noise(config["test_letter"], config["noise_level"], seed))

    network = HopfieldNetwork(stored_patterns)

    results = network.get_stored(test_pattern, max_epochs=10, detailed=True)

    for result in results:
        result["state"] = unflatten_pattern(result["state"])

    results_dir = Path("ej2_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = results_dir / f"output_{timestamp}"
    dir_path.mkdir(exist_ok=True)

    config_copy_path = dir_path / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=4)

    for letter in config["stored_letters"]:
        letter_image_path = dir_path / f"stored_{letter}.png"
        visualize_letter(patterns[letter], letter_image_path)

    test_letter_path = dir_path / f"test_{config['test_letter']}_noise_{config['noise_level']}.png"
    visualize_letter(unflatten_pattern(test_pattern), test_letter_path)

    final_state_path = dir_path / "final_state.png"
    visualize_letter(results[-1]["state"], final_state_path)

    results_path = dir_path / "results.json"
    with open(results_path, "w") as f:
        json.dump([result["energy"] for result in results], f, indent=4, cls=CompactRowsEncoder)

    animation_path = dir_path / "animation.mp4"
    process_animation([result["state"] for result in results], animation_path)


