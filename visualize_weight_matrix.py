import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define thoughtseeds and states for labeling
thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']

def visualize_weight_matrix(experience_level='novice'):
    # Load the weight matrix from the .json file
    json_file = f"learned_weights_{experience_level}.json"
    try:
        with open(json_file, 'r') as f:
            weight_matrix = np.array(json.load(f))
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return

    # Create a heatmap using seaborn for clear visualization
    plt.figure(figsize=(12, 8))
    sns_heatmap = sns.heatmap(
        weight_matrix,
        annot=True,  # Show numerical values in cells
        fmt='.3f',   # Format to 3 decimal places for precision
        cmap='viridis',  # Green-to-yellow colormap for positive weights, better for biological relevance
        xticklabels=states,  # Label x-axis with meditation states
        yticklabels=thoughtseeds,  # Label y-axis with thoughtseeds
        cbar_kws={'label': 'Weight Value'},  # Color bar label
        vmin=0.0, vmax=1.0,  # Adjust color scale to reflect expected biological range (0.05–1.0)
        annot_kws={"size": 12}  # Increase font size of weights
    )

    # Add title and labels with context for thoughtseed dynamics
    plt.title(f'Learned Weight Matrix for Thoughtseeds and States ({experience_level.capitalize()})', fontweight='bold', fontsize=16)
    plt.xlabel('Meditation State', fontweight='bold', fontsize=14)  # Increase font size to 14
    plt.ylabel('Thoughtseed', fontweight='bold', fontsize=14)  # Increase font size to 14
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save the plot with a unique name to avoid overwriting
    plt.savefig(f'./results/plots/weight_matrix_{experience_level}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weight matrix visualization saved as weight_matrix_{experience_level}.png")

    # Print summary statistics for biological plausibility and debugging
    print(f"\nWeight Matrix Summary for {experience_level}:")
    print(f"Shape: {weight_matrix.shape}")
    print(f"Mean weight: {np.mean(weight_matrix):.3f}")
    print(f"Max weight: {np.max(weight_matrix):.3f}")
    print(f"Min weight: {np.min(weight_matrix):.3f}")
    print(f"Std weight: {np.std(weight_matrix):.3f}")  # Added standard deviation for variability

if __name__ == "__main__":
    # Visualize for both novice and expert
    visualize_weight_matrix('novice')
    visualize_weight_matrix('expert')
