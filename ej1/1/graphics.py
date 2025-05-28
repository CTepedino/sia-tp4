import math
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon


def get_hex_coordinates(x, y):
    # Convert grid coordinates to hexagonal coordinates
    hex_x = x * 1.5
    hex_y = y * np.sqrt(3) + (x % 2) * np.sqrt(3) / 2
    return hex_x, hex_y

def plot_registers_heatmap_quad(directory, clusters, k):
    matrix = np.zeros((k, k), dtype=int)

    for (i, j), countries in clusters.items():
        matrix[i][j] = len(countries)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues", origin='lower')

    plt.title("Cantidad de Países por Neurona")
    plt.colorbar(label="Cantidad de Países")

    plt.xticks(np.arange(k))
    plt.yticks(np.arange(k))
    plt.grid(False)
    plt.savefig(directory / "heatmap.png")


def plot_registers_heatmap_hex(directory, clusters, k):
    matrix = np.zeros((k, k), dtype=int)

    for (i, j), countries in clusters.items():
        matrix[i][j] = len(countries)

    hex_radius = 1
    hex_height = np.sqrt(3) * hex_radius

    fig, ax = plt.subplots(figsize=(10, 10))

    max_count = np.max(matrix)
    for i in range(k):
        for j in range(k):
            x, y = get_hex_coordinates(i, j)
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = x + hex_radius * np.cos(angles)
            hex_y = y + hex_radius * np.sin(angles)

            count = matrix[i][j]
            color = plt.cm.Blues(count / max_count if max_count > 0 else 0)
            ax.fill(hex_x, hex_y, color=color, edgecolor='white')


    centers_x, centers_y = zip(*(get_hex_coordinates(i, j) for i in range(k) for j in range(k)))
    ax.set_xlim(min(centers_x) - 1, max(centers_x) + 1)
    ax.set_ylim(min(centers_y) - 1, max(centers_y) + 1)
    ax.set_aspect('equal')
    ax.axis('off')


    norm = plt.Normalize(vmin=0, vmax=max_count)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Cantidad de Países')

    ax.set_title("Cantidad de Países por Neurona")
    plt.savefig(directory / "heatmap.png")
    plt.close()

def plot_country_clusters_quad(directory, clusters, k):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Clusters de Países")

    for i in range(k):
        for j in range(k):
            ax.plot(j, i, 'ko')

    for (i, j), countries in clusters.items():
        for offset, country in enumerate(countries):
            ax.text(j + 0.05, i + 0.05 + 0.15 * offset, country, fontsize=9)

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(-0.5, k - 0.5)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(directory / "clusters.png")


def plot_country_clusters_hex(directory, clusters, k):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    plt.title("Clusters de Países")

    for i in range(k):
        for j in range(k):
            x, y = get_hex_coordinates(i, j)
            # Draw hexagon
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = x + np.cos(angles)
            hex_y = y + np.sin(angles)
            ax.plot(hex_x, hex_y, 'k-', alpha=0.3, linewidth=0.5)

    for (x, y), countries_list in clusters.items():
        hex_x, hex_y = get_hex_coordinates(x, y)
        plt.plot(hex_x, hex_y, 'ko', markersize=10)

        n_countries = len(countries_list)
        for i, country in enumerate(countries_list):
            angle = 2 * np.pi * i / n_countries
            offset = 0.3
            text_x = hex_x + offset * np.cos(angle)
            text_y = hex_y + offset * np.sin(angle)
            plt.text(text_x, text_y, country, fontsize=9, ha='center', va='center')


    hex_radius = 1
    centers_x = []
    centers_y = []
    for i in range(k):
        for j in range(k):
            x, y = get_hex_coordinates(i, j)
            centers_x.append(x)
            centers_y.append(y)
    min_x, max_x = min(centers_x), max(centers_x)
    min_y, max_y = min(centers_y), max(centers_y)
    plt.xlim(min_x - hex_radius, max_x + hex_radius)
    plt.ylim(min_y - hex_radius, max_y + hex_radius)

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(directory / "clusters.png")
    plt.close()

def plot_u_matrix_quad(directory, u_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(u_matrix, cmap='bone_r', interpolation='nearest', origin='upper')
    plt.colorbar(label='Distancia')
    plt.xticks(np.arange(u_matrix.shape[1]))
    plt.yticks(np.arange(u_matrix.shape[0]))
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(directory / "u_matrix.png")


def plot_u_matrix_hex(directory, u_matrix):
    som_shape = u_matrix.shape
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            x, y = get_hex_coordinates(i, j)
            angles = np.linspace(0, 2 * np.pi, 7)[:-1]
            hex_x = x + np.cos(angles)
            hex_y = y + np.sin(angles)
            plt.fill(hex_x, hex_y, color=plt.cm.bone_r(u_matrix[i, j] / u_matrix.max()))

    hex_radius = 1
    centers_x = []
    centers_y = []
    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            x, y = get_hex_coordinates(i, j)
            centers_x.append(x)
            centers_y.append(y)
    min_x, max_x = min(centers_x), max(centers_x)
    min_y, max_y = min(centers_y), max(centers_y)
    plt.xlim(min_x - hex_radius, max_x + hex_radius)
    plt.ylim(min_y - hex_radius, max_y + hex_radius)
    ax.axis('off')

    norm = plt.Normalize(0, u_matrix.max())
    sm = plt.cm.ScalarMappable(cmap='bone_r', norm=norm)
    plt.colorbar(sm, ax=plt.gca(), label='Distancia')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(directory / "u_matrix.png")

