import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from kohonen import Kohonen
import json
import argparse
import os

def min_max_normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def get_hex_coordinates(x, y):
    # Convert grid coordinates to hexagonal coordinates
    hex_x = x * 1.5
    hex_y = y * np.sqrt(3) + (x % 2) * np.sqrt(3) / 2
    return hex_x, hex_y

def plot_hex_grid(ax, som_shape):
    # Draw hexagonal grid lines
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            x, y = get_hex_coordinates(i, j)
            # Draw hexagon
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = x + np.cos(angles)
            hex_y = y + np.sin(angles)
            ax.plot(hex_x, hex_y, 'k-', alpha=0.3, linewidth=0.5)

parser = argparse.ArgumentParser(description='Análisis de países usando SOM')
parser.add_argument('config_file', type=str, help='Ruta al archivo de configuración JSON')
args = parser.parse_args()

try:
    with open(args.config_file, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de configuración '{args.config_file}'")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: El archivo '{args.config_file}' no es un JSON válido")
    exit(1)

df = pd.read_csv('europe.csv')
countries = df["Country"].values
data = df.drop(columns=["Country"]).values
data_norm = min_max_normalize(data)

som_shape = (config['som_shape']['width'], config['som_shape']['height'])
kohonen = Kohonen(som_shape[0], som_shape[1], input_len=data.shape[1], 
                 learning_rate=config['learning_rate'],
                 grid_type=config['grid_type'])
kohonen.train(data_norm, num_iterations=config['iterations'])

# Crear directorio results si no existe
results_dir = f"results_kohonen/{config['learning_rate']}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

mapped = kohonen.map_input(data_norm)

plt.figure(figsize=(12, 8))
ax = plt.gca()

neuron_groups = {}
for i, (x, y) in enumerate(mapped):
    if (x, y) not in neuron_groups:
        neuron_groups[(x, y)] = []
    neuron_groups[(x, y)].append(countries[i])

if config['grid_type'] == "hex":
    # Plot hexagonal grid
    plot_hex_grid(ax, som_shape)
    
    for (x, y), countries_list in neuron_groups.items():
        hex_x, hex_y = get_hex_coordinates(x, y)
        plt.plot(hex_x, hex_y, 'ko', markersize=10)
        
        n_countries = len(countries_list)
        for i, country in enumerate(countries_list):
            angle = 2 * np.pi * i / n_countries
            offset = 0.3
            text_x = hex_x + offset * np.cos(angle)
            text_y = hex_y + offset * np.sin(angle)
            plt.text(text_x, text_y, country, fontsize=9, ha='center', va='center')
    
    # Set appropriate limits for hexagonal grid
    max_x = som_shape[0] * 1.5
    max_y = som_shape[1] * np.sqrt(3) + np.sqrt(3) / 2
    plt.xlim(-0.5, max_x + 0.5)
    plt.ylim(-0.5, max_y + 0.5)
else:
    # Original square grid plotting
    for (x, y), countries_list in neuron_groups.items():
        plt.plot(x, y, 'ko', markersize=10)
        
        n_countries = len(countries_list)
        for i, country in enumerate(countries_list):
            angle = 2 * np.pi * i / n_countries
            offset = 0.2
            text_x = x + offset * np.cos(angle)
            text_y = y + offset * np.sin(angle)
            plt.text(text_x, text_y, country, fontsize=9, ha='center', va='center')
    
    plt.xlim(-0.5, som_shape[0]-0.5)
    plt.ylim(-0.5, som_shape[1]-0.5)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.title("Clusters de Países - Red de Kohonen")
plt.xlabel("X")
plt.ylabel("Y")

if config['grid_type'] == "quad":
    plt.xticks(np.arange(som_shape[0]))
    plt.yticks(np.arange(som_shape[1]))

plt.savefig(os.path.join(results_dir, 'country_clusters.png'))
plt.close()

def compute_u_matrix(model):
    umatrix = np.zeros((model.x, model.y))
    for i in range(model.x):
        for j in range(model.y):
            neighbors = model._neighborhood((i, j), radius=1.5)
            dists = [np.linalg.norm(model.weights[i, j] - model.weights[n_i, n_j])
                     for (n_i, n_j) in neighbors if (n_i, n_j) != (i, j)]
            umatrix[i, j] = np.mean(dists) if dists else 0
    return umatrix

u_matrix = compute_u_matrix(kohonen)

plt.figure(figsize=(10, 8))
if config['grid_type'] == "hex":
    # Create hexagonal U-matrix visualization
    hex_u_matrix = np.zeros((som_shape[0], som_shape[1]))
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            hex_x, hex_y = get_hex_coordinates(i, j)
            hex_u_matrix[i, j] = u_matrix[i, j]
    
    # Plot hexagonal U-matrix
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            x, y = get_hex_coordinates(i, j)
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = x + np.cos(angles)
            hex_y = y + np.sin(angles)
            plt.fill(hex_x, hex_y, color=plt.cm.bone_r(u_matrix[i, j]/u_matrix.max()))
    
    norm = plt.Normalize(0, u_matrix.max())
    sm = plt.cm.ScalarMappable(cmap='bone_r', norm=norm)
    plt.colorbar(sm, ax=plt.gca(), label='Distancia')
    max_x = som_shape[0] * 1.5
    max_y = som_shape[1] * np.sqrt(3) + np.sqrt(3) / 2
    plt.xlim(-0.5, max_x + 0.5)
    plt.ylim(-0.5, max_y + 0.5)
else:
    plt.imshow(u_matrix.T, cmap='bone_r', origin='lower')
    plt.colorbar(label='Distancia')

plt.title("U-Matrix (Distancia entre neuronas vecinas)")
plt.savefig(os.path.join(results_dir, 'u_matrix.png'))
plt.close()

frequencies = Counter(mapped)
heatmap = np.zeros(som_shape)
for (x, y), count in frequencies.items():
    heatmap[x, y] = count

plt.figure(figsize=(10, 8))
if config['grid_type'] == "hex":
    # Create hexagonal heatmap visualization
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            x, y = get_hex_coordinates(i, j)
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = x + np.cos(angles)
            hex_y = y + np.sin(angles)
            plt.fill(hex_x, hex_y, color=plt.cm.Blues(heatmap[i, j]/heatmap.max()))
    
    norm = plt.Normalize(0, heatmap.max())
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    plt.colorbar(sm, ax=plt.gca(), label='Cantidad de países')
    max_x = som_shape[0] * 1.5
    max_y = som_shape[1] * np.sqrt(3) + np.sqrt(3) / 2
    plt.xlim(-0.5, max_x + 0.5)
    plt.ylim(-0.5, max_y + 0.5)
else:
    plt.imshow(heatmap.T, cmap='Blues', origin='lower')
    plt.colorbar(label='Cantidad de países')

plt.title("Cantidad de Países por Neurona")
plt.savefig(os.path.join(results_dir, 'neuron_counts.png'))
plt.close()

print("\nAnálisis de Grupos de Países:")
print("----------------------------")
for (x, y), count in frequencies.items():
    countries_in_neuron = [countries[i] for i, (mx, my) in enumerate(mapped) if (mx, my) == (x, y)]
    print(f"\nNeurona ({x}, {y}) contiene {count} países:")
    print(", ".join(countries_in_neuron))

# Guardar resultados en un archivo ordenado
with open(os.path.join(results_dir, 'neuron_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Análisis de Grupos de Países:\n")
    f.write("----------------------------\n\n")
    
    # Iterar por orden de neuronas
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            if (i, j) in frequencies:
                countries_in_neuron = [countries[idx] for idx, (mx, my) in enumerate(mapped) if (mx, my) == (i, j)]
                f.write(f"Neurona ({i}, {j}) contiene {len(countries_in_neuron)} países:\n")
                f.write(", ".join(countries_in_neuron) + "\n\n") 