import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from kohonen import Kohonen

def min_max_normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)

# 1. Cargar y preparar los datos
df = pd.read_csv('europe.csv')
countries = df["Country"].values
data = df.drop(columns=["Country"]).values
data_norm = min_max_normalize(data)

# 2. Crear y entrenar la red de Kohonen
som_shape = (5, 5)  # Grid de 5x5 para mejor visualización
kohonen = Kohonen(som_shape[0], som_shape[1], input_len=data.shape[1], learning_rate=0.5)
kohonen.train(data_norm, num_iterations=10000)

# 3. Obtener las asignaciones de países a neuronas
mapped = kohonen.map_input(data_norm)

# 4. Visualización de clusters de países
plt.figure(figsize=(12, 8))

# Crear un diccionario para agrupar países por neurona
neuron_groups = {}
for i, (x, y) in enumerate(mapped):
    if (x, y) not in neuron_groups:
        neuron_groups[(x, y)] = []
    neuron_groups[(x, y)].append(countries[i])

# Dibujar el grid y los países
for (x, y), countries_list in neuron_groups.items():
    # Dibujar el punto central de la neurona
    plt.plot(x, y, 'ko', markersize=10)
    
    # Calcular el ángulo para distribuir los países alrededor del punto
    n_countries = len(countries_list)
    for i, country in enumerate(countries_list):
        angle = 2 * np.pi * i / n_countries
        offset = 0.2  # Distancia del punto central
        text_x = x + offset * np.cos(angle)
        text_y = y + offset * np.sin(angle)
        plt.text(text_x, text_y, country, fontsize=9, ha='center', va='center')

# Configurar los límites del gráfico
plt.xlim(-0.5, som_shape[0]-0.5)
plt.ylim(-0.5, som_shape[1]-0.5)

# Agregar grid y etiquetas
plt.grid(True, linestyle='--', alpha=0.7)
plt.title("Clusters de Países - Red de Kohonen")
plt.xlabel("X")
plt.ylabel("Y")

# Ajustar los ticks para que coincidan con las posiciones de las neuronas
plt.xticks(np.arange(som_shape[0]))
plt.yticks(np.arange(som_shape[1]))

plt.savefig('country_clusters.png')
plt.close()

# 5. U-Matrix (distancias promedio entre neuronas vecinas)
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
plt.imshow(u_matrix.T, cmap='bone_r', origin='lower')
plt.colorbar(label='Distancia')
plt.title("U-Matrix (Distancia entre neuronas vecinas)")
plt.savefig('u_matrix.png')
plt.close()

# 6. Análisis de cantidad de países por neurona
frequencies = Counter(mapped)
heatmap = np.zeros(som_shape)
for (x, y), count in frequencies.items():
    heatmap[x, y] = count

plt.figure(figsize=(10, 8))
plt.imshow(heatmap.T, cmap='Blues', origin='lower')
plt.colorbar(label='Cantidad de países')
plt.title("Cantidad de Países por Neurona")
plt.savefig('neuron_counts.png')
plt.close()

# 7. Imprimir análisis de grupos
print("\nAnálisis de Grupos de Países:")
print("----------------------------")
for (x, y), count in frequencies.items():
    countries_in_neuron = [countries[i] for i, (mx, my) in enumerate(mapped) if (mx, my) == (x, y)]
    print(f"\nNeurona ({x}, {y}) contiene {count} países:")
    print(", ".join(countries_in_neuron)) 