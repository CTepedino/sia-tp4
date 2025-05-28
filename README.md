# TP4 SIA - Aprendizaje No Supervisado

## Requisitos

- Python 3.8 o superior
- uv: gestor de entornos y dependencias

### Instalación de uv:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### Sincronización de dependencias:

Este proyecto incluye un archivo requirements.txt. Para instalar todo con uv, ejecutar:

```bash
uv venv .venv  
uv sync
```

Esto crea el entorno virtual en .venv e instala automáticamente las dependencias necesarias.

---

## Ejecución

Para ejecutar la red de Kohonen:

```bash
uv run ej1/1/main.py config_file_path
```

Para ejecutar la red de Oja:

```bash
uv run ej1/2/main.py config_file_path
```

Para ejecutar la red de Hopfield:

```bash
uv run ej2/main.py config_file_path
```

---
## Configuración

Los archivos de configuración se leen en formato JSON

El nivel ("level") se describe como una lista de strings, en donde cada string representa una fila del tablero.

Para la red de Kohonen:

- geometry: "quad" para geometria cuadrada o "hex" para geometria hexagonal
- k: lado de la grilla de neuronas de salida
- initial_r: radio inicial de vecinos
- distance_fn: función para el calculo de distancias en la red. Puede ser "euclidean" o "exponential"
- initial_lr: learning rate inicial
- iterations_per_n: cantidad de iteraciones en función de la dimensión de la entrada
- seed: semilla a usar por el randomizador

Para la red de Oja:

- epochs: cantidad de epocas del aprendizaje
- initial_lr: valor inicial del learning rate
- learning_rate: "constant" o "decaying"
- seed: semilla a usar por el randomizador

Para la red de Hopfield:

- stored_letters: arreglo con los patrones de letras que aprenderá la red
- test_letter: letra a probar
- noise_level: cantidad de ruido (en pixeles) a aplicar en la letra a probar