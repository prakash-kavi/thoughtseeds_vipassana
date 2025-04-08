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

def plot_meditation_dominant(results):
    time_steps = np.arange(results['timesteps'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Improved color palette
    thoughtseed_colors = {
        'self_reflection': '#4363d8',      # Blue
        'breath_focus': '#f58231',         # Orange
        'equanimity': '#3cb44b',           # Green
        'pain_discomfort': '#e6194B',      # Red
        'pending_tasks': '#911eb4'         # Purple
    }
    
    # Meditation States (top subplot)
    reordered_states = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
    state_colors = {
        "breath_control": "#7fcdbb",
        "mind_wandering": "#2c7fb8",
        "meta_awareness": "#f0ad4e", 
        "redirect_breath": "#5ab4ac"
    }
    
    # Get state transitions for highlighting
    state_history = results['state_history']
    state_transitions = [i for i in range(1, len(state_history)) 
                       if state_history[i] != state_history[i-1]]
    
    # Plot state transitions as a more elegant step function with filled areas
    state_indices = [reordered_states.index(state) for state in state_history]
    ax1.step(time_steps, state_indices, where='post', color='#444444', linewidth=1.5, alpha=0.7, label='State Transition')
    
    # Fill each state area with appropriate color
    for i in range(len(reordered_states)):
        # Create mask for this state level
        mask = np.array(state_indices) == i
        # Fill area under the line segment
        if any(mask):
            ax1.fill_between(time_steps, i, i+1, where=mask, step='post',
                           color=state_colors[reordered_states[i]], alpha=0.3)
    
    ax1.set_yticks(range(len(reordered_states)))
    ax1.set_yticklabels(reordered_states)
    ax1.set_title(f'{results["experience_level"].capitalize()}: Meditation States', fontsize=13, fontweight='bold')
    ax1.set_ylabel('State', fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add subtle background shading for state transitions
    for st in state_transitions:
        ax1.axvspan(st-0.5, st+0.5, color='#f0f0f0', alpha=0.9, zorder=0)
        ax2.axvspan(st-0.5, st+0.5, color='#f0f0f0', alpha=0.9, zorder=0)
    
    # Dominant Thoughtseeds (bottom subplot)
    reordered_thoughtseeds = ['self_reflection', 'breath_focus', 'equanimity', 'pain_discomfort', 'pending_tasks']
    ts_mapping = {ts: i for i, ts in enumerate(reversed(reordered_thoughtseeds))}
    dominant_ts_history = results['dominant_ts_history']
    
    # Use scatter points but connect with lines to show flow
    scatter_data = {}
    for i, ts in enumerate(dominant_ts_history):
        color = thoughtseed_colors[ts]
        idx = ts_mapping[ts]
        y_pos = 4 - idx
        if ts not in scatter_data:
            scatter_data[ts] = {'x': [], 'y': []}
        scatter_data[ts]['x'].append(time_steps[i])
        scatter_data[ts]['y'].append(y_pos)
    
    # Plot each thoughtseed's appearances
    for ts in reordered_thoughtseeds:
        if ts in scatter_data and scatter_data[ts]['x']:
            ax2.scatter(scatter_data[ts]['x'], scatter_data[ts]['y'], 
                       s=30, marker='o', facecolors=thoughtseed_colors[ts], edgecolors='white',
                       linewidth=0.5, alpha=0.9, label=ts, zorder=5)
    
    # Connect successive points with lines for better tracking
    prev_y = 4 - ts_mapping[dominant_ts_history[0]]
    for i in range(1, len(dominant_ts_history)):
        curr_y = 4 - ts_mapping[dominant_ts_history[i]]
        if dominant_ts_history[i] != dominant_ts_history[i-1]:
            ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        prev_y = curr_y
    
    # Identify successful and failed broadcasts
    
    # SUCCESSFUL BROADCASTS: thoughtseed changes that lead to state transitions
    successful_broadcasts = [i for i in range(1, len(dominant_ts_history)-1) 
                           if dominant_ts_history[i] != dominant_ts_history[i-1] and
                           i+1 in state_transitions]
    
    # Also identify sustained dominance leading to state changes
    for i in range(len(dominant_ts_history)-3):
        curr_ts = dominant_ts_history[i]
        if (dominant_ts_history[i+1] == curr_ts and 
            dominant_ts_history[i+2] == curr_ts and
            i+2 in state_transitions):
            successful_broadcasts.append(i+1)
    
    # FAILED BROADCASTS: thoughtseed changes that don't lead to state transitions
    ts_changes = [i for i in range(1, len(dominant_ts_history)) 
                 if dominant_ts_history[i] != dominant_ts_history[i-1]]
    
    failed_broadcasts = [i for i in ts_changes if i not in successful_broadcasts and i+1 not in successful_broadcasts]
    
    # Draw vertical lines showing successful broadcasts (causal relationships)
    # For successful broadcasts, use triangle markers with higher visibility
    for sb in successful_broadcasts:
        dominant_ts = dominant_ts_history[sb]
        ts_color = thoughtseed_colors[dominant_ts]
        y_pos = 4 - ts_mapping[dominant_ts]
        
        # Create colored triangle pointing up (causal effect)
        ax2.scatter(sb, y_pos, s=200, marker='^', facecolors=ts_color, 
                edgecolors='white', linewidth=1.5, alpha=0.9, zorder=15)
        
        # Add text label for clarity
        ax2.text(sb, y_pos-0.4, "BROADCAST", fontsize=7, 
                ha='center', color=ts_color, fontweight='bold', alpha=0.7)

    # For failed broadcasts, use square markers
    for fb in failed_broadcasts:
        dominant_ts = dominant_ts_history[fb]
        ts_color = thoughtseed_colors[dominant_ts]
        y_pos = 4 - ts_mapping[dominant_ts]
        
        # Create outlined square (ignition without causal effect)
        ax2.scatter(fb, y_pos, s=150, marker='s', facecolors='none',
                edgecolors=ts_color, linewidth=2.0, alpha=0.7, zorder=15)
    
    ax2.set_yticks(range(len(reordered_thoughtseeds)))
    ax2.set_yticklabels(reordered_thoughtseeds)
    ax2.invert_yaxis()
    ax2.set_title(f'{results["experience_level"].capitalize()}: Dominant Thoughtseeds', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Thoughtseed', fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Improved legend
    legend_items = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=8, label=ts) 
                   for ts, color in thoughtseed_colors.items()]
    
    # Add broadcast markers to legend
    legend_items.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='none', 
                                 markeredgecolor='black', markersize=10, 
                                 label='Successful Broadcast'))
    
    legend_items.append(plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='none', 
                                 markeredgecolor='gray', markersize=8, 
                                 label='Failed Broadcast'))
    
    legend = ax2.legend(handles=legend_items, loc='upper right', 
                       bbox_to_anchor=(1.15, 1), framealpha=0.9,
                       fancybox=True, shadow=True, fontsize=10)
    
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#dddddd')
    
    # Add explanatory annotation about the broadcasts
    plt.figtext(0.01, 0.01, 
              "★ Successful Broadcasts: When thoughtseeds propagate to influence state transitions\n"
              "✗ Failed Broadcasts: Thoughtseed ignitions that don't affect meditation state", 
              fontsize=9, style='italic', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)  # Reduce space between plots
    plt.savefig(f'./results/plots/simulation_{results["experience_level"]}_intrinsic_ignition.png', 
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
            plot_meditation_dominant(results)
            print(f"Plots for {experience_level} saved to ./results/plots/")
        else:
            print(f"Skipping plot generation for {experience_level} due to missing data")