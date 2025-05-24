import numpy as np
import random
import copy
import argparse
import json
import sys

# Definición de patrones de letras (5x5, 1 y -1)
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

# Función para mostrar una letra
def print_pattern(pattern):
    for row in pattern:
        print(' '.join(['*' if x == 1 else ' ' for x in row]))
    print()

# Hopfield Network
class HopfieldNet:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            v = p.flatten()
            self.W += np.outer(v, v)
        np.fill_diagonal(self.W, 0)
        self.W /= len(patterns)

    def recall(self, pattern, steps=10, verbose=True):
        v = pattern.flatten()
        for step in range(steps):
            v_new = np.sign(self.W @ v)
            v_new[v_new == 0] = 1
            if verbose:
                print(f"Paso {step+1}:")
                print_pattern(v_new.reshape((5,5)))
            if np.array_equal(v, v_new):
                break
            v = v_new
        return v.reshape((5,5))

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
    parser.add_argument('config', nargs='?', default=None, help='Ruta a archivo JSON de configuración (opcional)')
    parser.add_argument('--test_letter', type=str, default='J', help='Letra a testear (J, H, L, E)')
    parser.add_argument('--noise_level', type=float, default=0.3, help='Nivel de ruido (0.0 a 1.0)')
    args = parser.parse_args()

    # Si se pasa un archivo de configuración, leer de ahí
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        test_letter = config.get('test_letter', 'J').upper()
        noise_level = config.get('noise_level', 0.3)
    else:
        test_letter = args.test_letter.upper()
        noise_level = args.noise_level

    print("Patrones originales:")
    for name, pat in patterns.items():
        print(f"Letra {name}:")
        print_pattern(pat)

    net = HopfieldNet(25)
    net.train([p for p in patterns.values()])

    if test_letter not in patterns:
        raise ValueError(f"Letra de prueba '{test_letter}' no está entre las letras disponibles: {list(patterns.keys())}")

    noisy_pattern = add_noise(patterns[test_letter], noise_level)
    print(f"Patrón ruidoso de la letra {test_letter} (ruido {int(noise_level*100)}%):")
    print_pattern(noisy_pattern)

    # Recuperar el patrón usando la red de Hopfield
    print("Evolución de la recuperación:")
    recovered = net.recall(noisy_pattern, steps=10, verbose=True)
    print("Patrón recuperado:")
    print_pattern(recovered) 