import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Set global matplotlib styling for a cleaner, more modern look
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3

# Rest of imports and constants remain the same...

def plot_hierarchy(results):
    time_steps = np.arange(results['timesteps'])
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.5]})
    
    # Improved color palette
    thoughtseed_colors = {
        'self_reflection': '#4363d8',      # Blue
        'breath_focus': '#f58231',         # Orange
        'equanimity': '#3cb44b',           # Green
        'pain_discomfort': '#e6194B',      # Red
        'pending_tasks': '#911eb4'         # Purple
    }
    
    # Add a subtle background shade for state transitions
    state_changes = [i for i in range(1, len(results['state_history'])) 
                     if results['state_history'][i] != results['state_history'][i-1]]
    for sc in state_changes:
        ax1.axvspan(sc-0.5, sc+0.5, color='#f0f0f0', alpha=0.7, zorder=0)
        ax2.axvspan(sc-0.5, sc+0.5, color='#f0f0f0', alpha=0.7, zorder=0)
        ax3.axvspan(sc-0.5, sc+0.5, color='#f0f0f0', alpha=0.7, zorder=0)

    # Level 3: Meta-Cognition with improved aesthetics
    smoothed_meta = np.zeros_like(results['meta_awareness_history'])
    alpha = 0.3
    smoothed_meta[0] = results['meta_awareness_history'][0]
    for j in range(1, len(results['meta_awareness_history'])):
        smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * results['meta_awareness_history'][j]
    
    # Fill between line and baseline for better visibility
    ax1.plot(time_steps, smoothed_meta, color=thoughtseed_colors['self_reflection'], linewidth=2.5)
    ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color=thoughtseed_colors['self_reflection'])
    ax1.set_ylabel('Meta-Awareness', fontsize=11)
    ax1.set_title(f'Level 3: Meta-Cognition ({results["experience_level"].capitalize()})', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Level 2: Dominant Thoughtseed with improved aesthetics
    reordered_thoughtseeds = ['self_reflection', 'breath_focus', 'equanimity', 'pain_discomfort', 'pending_tasks']
    ts_mapping = {ts: i for i, ts in enumerate(reversed(reordered_thoughtseeds))}
    
    # Use larger, more visible dots for dominant thoughtseeds
    for i, ts in enumerate(results['dominant_ts_history']):
        color = thoughtseed_colors[ts]
        idx = ts_mapping[ts]
        ax2.scatter(time_steps[i], 4 - idx, s=30, marker='o', facecolors=color, edgecolors='white', linewidth=0.5, alpha=0.9, zorder=5)
    
    # Connect dots with thin lines for better tracking
    prev_y = 4 - ts_mapping[results['dominant_ts_history'][0]]
    for i in range(1, len(results['dominant_ts_history'])):
        curr_y = 4 - ts_mapping[results['dominant_ts_history'][i]]
        if results['dominant_ts_history'][i] != results['dominant_ts_history'][i-1]:
            ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        prev_y = curr_y
        
    ax2.set_yticks(range(len(reordered_thoughtseeds)))
    ax2.set_yticklabels(reordered_thoughtseeds)
    ax2.invert_yaxis()
    ax2.set_ylabel('Dominant Thoughtseed', fontsize=11)
    ax2.set_title(f'Level 2: Dominant Thoughtseed ({results["experience_level"].capitalize()})', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Level 1: Thoughtseed Activations with improved aesthetics
    for ts in reordered_thoughtseeds:
        i = results['thoughtseeds'].index(ts)
        activations = [act[i] for act in results['activations_history']]
        smoothed_activations = np.zeros_like(activations)
        alpha = 0.3
        smoothed_activations[0] = activations[0]
        for j in range(1, len(activations)):
            smoothed_activations[j] = (1 - alpha) * smoothed_activations[j-1] + alpha * activations[j]
        ax3.plot(time_steps, smoothed_activations, label=ts, color=thoughtseed_colors[ts], linewidth=2)
    
    ax3.set_xlabel('Timestep (t)', fontsize=11)
    ax3.set_ylabel('Activation', fontsize=11)
    ax3.set_title(f'Level 1: Thoughtseed Activations ({results["experience_level"].capitalize()})', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Create a more elegant legend with custom styling
    legend = ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1), framealpha=0.9, 
                        fancybox=True, shadow=True, fontsize=10)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#dddddd')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)  # Reduce space between plots
    plt.savefig(f'./results/plots/simulation_{results["experience_level"]}_hierarchy.png', 
               facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')
    plt.close()
    
def load_simulation_results(experience_level):
    """Load simulation results from pickle file"""
    try:
        with open(f"./results/data/simulation_results_{experience_level}.pkl", "rb") as f:
            results = pickle.load(f)
        print(f"Loaded simulation results for {experience_level}")
        return results
    except FileNotFoundError:
        print(f"Error: Simulation results not found for {experience_level}.")
        print(f"Make sure ./results/data/simulation_results_{experience_level}.pkl exists.")
        return None

def ensure_output_directories():
    """Create output directories if they don't exist"""
    os.makedirs('./results/plots', exist_ok=True)
    print("Output directory ./results/plots verified")
        
if __name__ == "__main__":
    # Create output directories
    ensure_output_directories()
    
    # Process both experience levels
    for experience_level in ['novice', 'expert']:
        # Load simulation results
        results = load_simulation_results(experience_level)
        
        if results:
            print(f"Creating plots for {experience_level}...")
            # Generate plots
            plot_hierarchy(results)
            print(f"Plots for {experience_level} saved to ./results/plots/")
        else:
            print(f"Skipping plot generation for {experience_level} due to missing data")
