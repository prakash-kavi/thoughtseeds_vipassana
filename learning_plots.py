"""
learning_plots.py

This file provides functions to visualize the results of the meditation learning simulation for 200 timesteps. 
It generates plots that illustrate key aspects of the learning process, aiding in the analysis of state transitions, 
meta-awareness, and thoughtseed activation dynamics.

### Inputs:
- `self.state_history`: A list of meditation states (`breath_control`, `mind_wandering`, `meta_awareness`, `redirect_breath`) recorded at each timestep.
- `self.meta_awareness_history`: A list of meta-awareness levels corresponding to each timestep.
- `self.activations_history`: A list of activation levels for each thoughtseed (`self_reflection`, `breath_focus`, `equanimity`, `pain_discomfort`, `pending_tasks`) across all timesteps.
- `self.timesteps`: Total number of timesteps (default: 200).
- `self.thoughtseeds`: List of thoughtseeds used in the simulation.
- `self.state_indices`: A mapping of state names to integer indices for visualization.

### Outputs:
- **Meditation States Plot (Figure 6)**:
  - A PNG file visualizing the transitions between meditation states over time, saved as:
    `./results/plots/learning_{experience_level}_meditation_states.png`.

- **Meta-awareness and Thoughtseed Activations Plot (Figure 5)**:
  - A PNG file containing two subplots:
    1. Smoothed meta-awareness levels over time.
    2. Smoothed activation levels for all thoughtseeds over time.
  - Saved as: `./results/plots/learning_{experience_level}_thoughtseed_meta_activations.png`.

"""
import matplotlib.pyplot as plt
import numpy as np

def plot_results(self):
    # Plot meditation states, thoughtseed activations, and meta-awareness with consistent axes
    time_steps = np.arange(self.timesteps)
    
    # Improved color palette
    thoughtseed_colors = {
        'self_reflection': '#4363d8',      # Blue
        'breath_focus': '#f58231',         # Orange
        'equanimity': '#3cb44b',           # Green
        'pain_discomfort': '#e6194B',      # Red
        'pending_tasks': '#911eb4'         # Purple
    }
    
    # State colors for better visualization
    state_colors = {
        "breath_control": "#7fcdbb",
        "mind_wandering": "#2c7fb8",
        "meta_awareness": "#f0ad4e", 
        "redirect_breath": "#5ab4ac"
    }

    # Ensure state_indices is a list of integers
    state_indices = [self.state_indices[state] for state in self.state_history]
    
    # Plot meditation states with improved styling
    plt.figure(figsize=(12, 3))
    
    # Plot state transitions as step function with filled areas for each state
    plt.step(time_steps, state_indices, where='post', color='#0000FF', linewidth=1.5, alpha=0.7)
    
    states = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
    # for i, state in enumerate(states):
    #     mask = np.array(state_indices) == i
    #     if any(mask):
    #         plt.fill_between(time_steps, i, i+1, where=mask, step='post',
    #                        color=state_colors[state], alpha=0.3)
    
    plt.xlabel('Timestep', fontsize=11)
    plt.ylabel('State', fontsize=11)
    plt.yticks([0, 1, 2, 3], states)
    plt.title(f'Learning: Meditation States Over Time ({self.experience_level.capitalize()})', 
             fontsize=13, fontweight='bold')
    plt.xlim(0, self.timesteps)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f'./results/plots/learning_{self.experience_level}_meditation_states.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

    # Plot thoughtseed activations and meta-awareness with consistent ranges - SWAPPED ORDER
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Meta-awareness on top (ax1) with smoothing
    smoothed_meta = np.zeros_like(self.meta_awareness_history)
    alpha = 0.3
    smoothed_meta[0] = self.meta_awareness_history[0]
    for j in range(1, len(self.meta_awareness_history)):
        smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * self.meta_awareness_history[j]
    
    ax1.plot(time_steps, smoothed_meta, color=thoughtseed_colors['self_reflection'], linewidth=2.5, label='Meta-Awareness')
    ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color=thoughtseed_colors['self_reflection'])
    ax1.set_ylabel('Meta-Awareness', fontsize=11)
    ax1.set_title(f'Learning: Meta-Awareness ({self.experience_level.capitalize()})', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0.55, 1.0)  # Fixed y-axis for meta-awareness (0.55–1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Thoughtseed activations below (ax2)
    reordered_thoughtseeds = ['self_reflection', 'breath_focus', 'equanimity', 'pain_discomfort', 'pending_tasks']
    for ts in reordered_thoughtseeds:
        i = self.thoughtseeds.index(ts)
        activations = [act[i] for act in self.activations_history]
        smoothed_activations = np.zeros_like(activations)
        alpha = 0.3  # Smoothing for neural inertia
        smoothed_activations[0] = activations[0]
        for j in range(1, len(activations)):
            smoothed_activations[j] = (1 - alpha) * smoothed_activations[j-1] + alpha * activations[j]
        ax2.plot(time_steps, smoothed_activations, label=ts, color=thoughtseed_colors[ts], linewidth=2)
    
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Activation', fontsize=11)
    ax2.set_title(f'Learning: Thoughtseed Activations ({self.experience_level.capitalize()})', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0.0, 1.05)  # Fixed y-axis for activations
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, self.timesteps)

    plt.tight_layout()
    plt.savefig(f'./results/plots/learning_{self.experience_level}_thoughtseed_meta_activations.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
