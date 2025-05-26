import json
import matplotlib.pyplot as plt
import numpy as np

def plot_patterns(data):
    # Crear una figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Patrones de la letra {data["input_data"]["test_letter"]} (Ruido: {data["input_data"]["noise_level"]*100}%)')

    # Convertir los patrones a arrays de numpy
    original = np.array(data["original_pattern"])
    noisy = np.array(data["noisy_pattern"])
    final = np.array(data["final_pattern"])

    # Reshape a matrices 5x5
    original = original.reshape(5, 5)
    noisy = noisy.reshape(5, 5)
    final = final.reshape(5, 5)

    # Plotear cada patrón
    axes[0].imshow(original, cmap='binary')
    axes[0].set_title('Patrón Original')
    axes[0].axis('off')

    axes[1].imshow(noisy, cmap='binary')
    axes[1].set_title('Patrón con Ruido')
    axes[1].axis('off')

    axes[2].imshow(final, cmap='binary')
    axes[2].set_title('Patrón Recuperado')
    axes[2].axis('off')

    # Ajustar el layout
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig('results_hopfield/visualization.png')
    plt.close()

if __name__ == "__main__":
    # Leer el archivo JSON
    with open('results_hopfield/output.json', 'r') as f:
        data = json.load(f)
    
    # Visualizar los patrones
    plot_patterns(data)
    print("Visualización guardada en results_hopfield/visualization.png") 