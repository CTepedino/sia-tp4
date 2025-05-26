import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_animation(data):
    # Crear la figura
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f'Letra {data["input_data"]["test_letter"]} (Ruido: {data["input_data"]["noise_level"]*100}%)', fontsize=16)
    
    # Convertir los patrones a arrays de numpy
    original = np.array(data["original_pattern"]).reshape(5, 5)
    noisy = np.array(data["noisy_pattern"]).reshape(5, 5)
    final = np.array(data["final_pattern"]).reshape(5, 5)
    
    # Crear la secuencia de frames
    frames = [original, noisy]  # Comenzamos con el original y el ruidoso
    frames.extend([np.array(step).reshape(5, 5) for step in data["evolution_steps"]])
    frames.append(final)  # Aseguramos que el último frame sea el final
    
    # Función de inicialización
    def init():
        ax.clear()
        ax.axis('off')
        return []

    # Función de animación
    def animate(i):
        ax.clear()
        ax.axis('off')
        
        # Mostrar el patrón actual
        im = ax.imshow(frames[i], cmap='binary')
        
        # Agregar título según el frame
        if i == 0:
            title = "Patrón Original"
        elif i == 1:
            title = "Patrón con Ruido"
        elif i == len(frames) - 1:
            title = "Patrón Final"
        else:
            title = f"Paso {i-1}"
            
        ax.set_title(title, fontsize=12)
        return [im]

    # Crear la animación
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=len(frames),
        interval=1000,  # 1 segundo entre frames
        blit=True
    )
    
    # Guardar la animación
    anim.save('results_hopfield/animation.gif', writer='pillow', fps=1)
    plt.close()

if __name__ == "__main__":
    # Leer el archivo JSON
    with open('results_hopfield/output.json', 'r') as f:
        data = json.load(f)
    
    # Crear la animación
    create_animation(data)
    print("Animación guardada en results_hopfield/animation.gif") 