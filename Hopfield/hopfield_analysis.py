import numpy as np
import random
import argparse
import json
import sys
from hopfield import HopfieldNet

patterns = {
    'J': np.array([
        [ 1,  1,  1,  1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1, 1, -1],
    ]),
    'H': np.array([
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
    ]),
    'L': np.array([
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ]),
    'E': np.array([
        [ 1,  1,  1,  1,  1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ]),
}

def print_pattern(pattern):
    for row in pattern:
        print(' '.join(['*' if x == 1 else ' ' for x in row]))
    print()


def add_noise(pattern, noise_level=0.3):
    noisy = pattern.flatten().copy()
    n = len(noisy)
    num_noisy = int(noise_level * n)
    idx = random.sample(range(n), num_noisy)
    for i in idx:
        noisy[i] *= -1
    return noisy.reshape(pattern.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hopfield letter recognition")
    parser.add_argument('config', help='Ruta al archivo JSON de configuración')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        sys.exit(1)
    except json.JSONDecodeError:
        sys.exit(1)

    test_letter = config.get('test_letter', '').upper()
    noise_level = config.get('noise_level', 0.3)

    if not test_letter:
        sys.exit(1)

    if test_letter not in patterns:
        sys.exit(1)

    if not 0 <= noise_level <= 1:
        sys.exit(1)

    net = HopfieldNet(25)
    net.train([p for p in patterns.values()])

    noisy_pattern = add_noise(patterns[test_letter], noise_level)
    print(f"Patrón ruidoso de la letra {test_letter} (ruido {int(noise_level*100)}%):")
    print_pattern(noisy_pattern)

    print("Evolución de la recuperación:")
    recovered = net.recall(noisy_pattern, steps=10, verbose=True)
    print("Patrón recuperado:")
    print_pattern(recovered)

    # Save results to file
    results = {
        "input_data": {
            "test_letter": test_letter,
            "noise_level": noise_level
        },
        "original_pattern": patterns[test_letter].tolist(),
        "noisy_pattern": noisy_pattern.tolist(),
        "evolution_steps": [recovered.tolist() for _ in range(10)],
        "final_pattern": recovered.tolist()
    }
    output_path = config.get('output_path', 'results_hopfield/output.json')
    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"Resultados guardados en {output_path}") 