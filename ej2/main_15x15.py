import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from hopfield_network import HopfieldNetwork
from letters_15x15 import letters as patterns

def flatten_pattern(matrix):
    return [x for row in matrix for x in row]

def unflatten_pattern(array, rows=15, cols=15):
    return [array[i * cols:(i + 1) * cols] for i in range(rows)]

def pattern_with_noise(letter, flips: int):
    import random
    pattern = [row[:] for row in patterns[letter]]  

    positions = list(range(225))  
    flip_positions = random.sample(positions, flips)

    for pos in flip_positions:
        row = pos // 15
        col = pos % 15
        pattern[row][col] = -pattern[row][col]

    return pattern

def calculate_success_rate(original, recovered):
    matches = sum(1 for a, b in zip(flatten_pattern(original), flatten_pattern(recovered)) if a == b)
    return matches / 225  

if __name__ == "__main__":
   
    noise_level = 25
    num_tests = 10000
    all_letters = list(patterns.keys())
    
   
    stored_patterns = [flatten_pattern(patterns[letter]) for letter in all_letters]
    network = HopfieldNetwork(stored_patterns)

    
    results_dir = Path("ej2_results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = results_dir / f"output_{timestamp}"
    dir_path.mkdir(exist_ok=True)

   
    results_by_letter = {}

    for letter in all_letters:
        success_rates = []
       
        
        for test in range(num_tests):
            
            test_pattern = flatten_pattern(pattern_with_noise(letter, noise_level))
            
            
            result = network.get_stored(test_pattern, max_epochs=10, detailed=True)
            
            
            final_state = unflatten_pattern(result[-1]["state"])
            success_rate = calculate_success_rate(patterns[letter], final_state)
            success_rates.append(success_rate)
            
           

        
        results_by_letter[letter] = {
            "mean_success_rate": np.mean(success_rates),
            "std_success_rate": np.std(success_rates),
           
        }

    
    results_path = dir_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_by_letter, f, indent=4)

    
    plt.figure(figsize=(15, 6))
    letters = list(results_by_letter.keys())
    success_rates = [results_by_letter[letter]["mean_success_rate"] for letter in letters]
    std_rates = [results_by_letter[letter]["std_success_rate"] for letter in letters]

    plt.bar(letters, success_rates, yerr=std_rates, capsize=5)
    plt.title('Tasa de éxito por letra')
    plt.xlabel('Letra')
    plt.ylabel('Tasa de éxito')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(dir_path / 'success_rates.png')
    plt.close()

    
    print("\nResultados por letra:")
    print("Letra\tTasa de éxito\tDesviación")
    print("-" * 60)
    for letter in all_letters:
        stats = results_by_letter[letter]
        print(f"{letter}\t{stats['mean_success_rate']:.3f}\t\t{stats['std_success_rate']:.3f}") 