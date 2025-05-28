import matplotlib.pyplot as plt
import pandas as pd

def generate_pc1_graphics(directory, pc1_vector, pc1_results, countries):

    variable_names = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]

    plt.figure(figsize=(15, 8))
    pc1_scores = pd.DataFrame({'Country': countries, 'PC1': pc1_results})
    pc1_scores_sorted = pc1_scores.sort_values('Country')
    plt.bar(range(len(pc1_scores_sorted)), pc1_scores_sorted['PC1'])
    plt.xticks(range(len(pc1_scores_sorted)), pc1_scores_sorted['Country'], rotation=45, ha='right')
    plt.ylabel('PC1')
    plt.grid(True, axis='y')
    plt.ylim(-5, 5)
    plt.tight_layout()
    plt.savefig(directory / 'pc1_countries.png')
    plt.close()


    plt.figure(figsize=(12, 6))
    feature_contributions = pd.DataFrame(
        pc1_vector,
        index=variable_names,
        columns=['PC1']
    )
    feature_contributions_sorted = feature_contributions.sort_index()
    plt.bar(range(len(feature_contributions_sorted)), feature_contributions_sorted['PC1'])
    plt.xticks(range(len(feature_contributions_sorted)), feature_contributions_sorted.index, rotation=45, ha='right')
    plt.ylabel('Contribución a PC1')
    plt.grid(True, axis='y')
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(directory / 'pc1_contributions.png')
    plt.close()
