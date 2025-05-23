import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from kohonen import Kohonen  # asegurate que el archivo se llame kohonen.py y esté en el mismo directorio


def min_max_normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)


# 1. Cargar el CSV
df = pd.read_csv('europe.csv')

# 2. Extraer solo las features numéricas (no el país)
countries = df["Country"].values
data = df.drop(columns=["Country"]).values

data_norm = min_max_normalize(data)

som_shape = (6, 4)
kohonen = Kohonen(som_shape[0], som_shape[1], input_len=data.shape[1], learning_rate=0.5)
kohonen.train(data_norm, num_iterations=10000)

# 5. Graficar los países sobre el mapa SOM
mapped = kohonen.map_input(data_norm)

plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(mapped):
    plt.text(x + 0.3*np.random.rand(), y + 0.3*np.random.rand(), countries[i], fontsize=9)
plt.title("Mapa de Países - Red de Kohonen")
plt.grid()
plt.show()

# 6. U-Matrix (distancias promedio entre neuronas vecinas)
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

plt.figure(figsize=(8, 6))
plt.imshow(u_matrix.T, cmap='bone_r', origin='lower')
plt.colorbar(label='Distancia')
plt.title("U-Matrix (Distancia entre neuronas vecinas)")
plt.show()

# 7. Histograma de países por neurona
frequencies = Counter(mapped)
heatmap = np.zeros(som_shape)
for (x, y), count in frequencies.items():
    heatmap[x, y] = count

plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, cmap='Blues', origin='lower')
plt.colorbar(label='Cantidad de países')
plt.title("Frecuencia por Neurona")
plt.show()
