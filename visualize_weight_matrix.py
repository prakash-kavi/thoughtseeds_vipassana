import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# Define thoughtseeds and states for labeling
thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']

def floor_to_two_decimals(matrix):
    return np.vectorize(lambda x: math.floor(x * 100) / 100)(matrix)

def load_weight_matrix(experience_level):
    # Load the weight matrix from the .json file
    json_file = f"learned_weights_{experience_level}.json"
    try:
        with open(json_file, 'r') as f:
            weight_matrix = np.array(json.load(f))
        weight_matrix = floor_to_two_decimals(weight_matrix)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return

def plot_weight_matrices():
    novice_weight_matrix = load_weight_matrix('novice')
    expert_weight_matrix = load_weight_matrix('expert')

    if novice_weight_matrix is None or expert_weight_matrix is None:
        return

    fig, axs = plt.subplots(1, 2, figsize=(24, 8))

    sns.heatmap(
        expert_weight_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=states,
        yticklabels=thoughtseeds,
        cbar_kws={'label': 'Weight Value'},
        vmin=0.0, vmax=1.0,
        annot_kws={"size": 12},
        ax=axs[0]
    )
    axs[0].set_title('Learned Weight Matrix for Thoughtseeds and States (Expert)', fontweight='bold', fontsize=16)
    axs[0].set_xlabel('Meditation State', fontweight='bold', fontsize=14)
    axs[0].set_ylabel('Thoughtseed', fontweight='bold', fontsize=14)

    sns.heatmap(
        novice_weight_matrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=states,
        yticklabels=thoughtseeds,
        cbar_kws={'label': 'Weight Value'},
        vmin=0.0, vmax=1.0,
        annot_kws={"size": 12},
        ax=axs[1]
    )
    axs[1].set_title('Learned Weight Matrix for Thoughtseeds and States (Novice)', fontweight='bold', fontsize=16)
    axs[1].set_xlabel('Meditation State', fontweight='bold', fontsize=14)
    axs[1].set_ylabel('Thoughtseed', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('./results/plots/weight_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Weight matrices visualization saved as weight_matrices_comparison.png")

if __name__ == "__main__":
    plot_weight_matrices()
