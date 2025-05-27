import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_pc1_graphics(directory, pc1, countries, data):

    variable_names = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]

    # Calcular contribuciones directamente del pc1 recibido
    contributions = pc1 ** 2
    contributions /= contributions.sum()

    # Ordenar variables alfabéticamente (manteniendo contribuciones correspondientes)
    sorted_indices_vars = np.argsort(variable_names)
    sorted_variable_names = [variable_names[i] for i in sorted_indices_vars]
    sorted_contributions = [contributions[i] for i in sorted_indices_vars]

    # --- Gráfico 1: Contribución de variables al PC1 ---
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(sorted_variable_names))
    plt.barh(y_pos, sorted_contributions, color="skyblue")
    plt.yticks(y_pos, sorted_variable_names)
    plt.xlabel("Contribución (normalizada)")
    plt.title("Contribución de cada variable al PC1")
    plt.tight_layout()
    plt.savefig(directory / "variable_contribution_pc1.png")
    plt.close()

    # --- Gráfico 2: PC1 por país ---

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(countries))
    plt.barh(y_pos, pc1, color="lightgreen")
    plt.yticks(y_pos, countries)
    plt.xlabel("PC1")
    plt.title("PC1 por país")
    plt.tight_layout()
    plt.savefig(directory / "pc1_per_country.png")
    plt.close()
