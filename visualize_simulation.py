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

# Added missing function: ensure_output_directories()
def ensure_output_directories():
    """Create necessary directories for output files"""
    os.makedirs('./results/plots', exist_ok=True)
    os.makedirs('./results/data', exist_ok=True)
    print("Output directories created/verified")

# Added missing function: load_simulation_results()
def load_simulation_results(experience_level):
    """Load simulation results from pickle file"""
    try:
        # Changed file path pattern to match existing files
        filepath = f'./results/data/simulation_results_{experience_level}.pkl'
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded {experience_level} simulation results from {filepath}")
        return results
    except FileNotFoundError:
        print(f"Error: Could not find simulation results file for {experience_level}")
        print(f"Expected location: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading simulation results for {experience_level}: {str(e)}")
        return None

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

def plot_thoughtseed_broadcasting(results):
    """
    Visualizes meditation states and thoughtseed broadcasts using Global Workspace Theory framework.
    Uses state dwell times to better classify successful vs potential broadcasts.
    """
    from meditation_config import STATE_DWELL_TIMES  # Import state dwell times
    
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
    
    # Plot state transitions as a step function with filled areas
    state_indices = [reordered_states.index(state) for state in state_history]
    ax1.step(time_steps, state_indices, where='post', color='#444444', linewidth=1.5, alpha=0.7)
    
    # Fill each state area with appropriate color
    for i in range(len(reordered_states)):
        # Create mask for this state level
        mask = np.array(state_indices) == i
        # Fill area under the line segment
        if any(mask):
            ax1.fill_between(time_steps, i, i+1, where=mask, step='post',
                           color=state_colors[reordered_states[i]], alpha=0.3)
    
    # Identify state dwell periods and eligibility for transitions
    min_dwell_times = {state: STATE_DWELL_TIMES[results['experience_level']][state][0] 
                     for state in reordered_states}
    
    # Track state start times
    state_start_times = [0]  # First state starts at time 0
    state_start_times.extend(state_transitions)
    
    # Calculate eligibility windows (when a state has reached minimum dwell time)
    eligible_for_transition = np.zeros_like(time_steps, dtype=bool)
    
    for t_idx in range(len(time_steps)):
        # Find what state period this timestep belongs to
        state_period_idx = next((i for i, st in enumerate(state_transitions) if st > t_idx), len(state_transitions))
        state_start = state_start_times[state_period_idx]
        current_state = state_history[t_idx]
        
        # Check if enough time has elapsed since state start
        elapsed = t_idx - state_start
        if elapsed >= min_dwell_times[current_state]:
            eligible_for_transition[t_idx] = True
    
    # Add eligibility shading to indicate when transitions are possible
    for i in range(1, len(time_steps)):
        if eligible_for_transition[i] and not eligible_for_transition[i-1]:
            # Start of eligibility window
            ax1.axvspan(i-0.5, i+0.5, color='#ffcccb', alpha=0.2, zorder=1)
    
    # Add subtle background shading for state transitions
    for st in state_transitions:
        ax1.axvspan(st-0.5, st+0.5, color='#f0f0f0', alpha=0.9, zorder=0)
        ax2.axvspan(st-0.5, st+0.5, color='#f0f0f0', alpha=0.9, zorder=0)
    
    ax1.set_yticks(range(len(reordered_states)))
    ax1.set_yticklabels(reordered_states)
    ax1.set_title(f'{results["experience_level"].capitalize()}: Meditation States and Broadcasts', 
                 fontsize=13, fontweight='bold')
    ax1.set_ylabel('State', fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
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
    
    # Connect successive points with lines
    prev_y = 4 - ts_mapping[dominant_ts_history[0]]
    for i in range(1, len(dominant_ts_history)):
        curr_y = 4 - ts_mapping[dominant_ts_history[i]]
        if dominant_ts_history[i] != dominant_ts_history[i-1]:
            ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        prev_y = curr_y
    
    # Identify thoughtseed changes
    ts_changes = [i for i in range(1, len(dominant_ts_history)) 
                 if dominant_ts_history[i] != dominant_ts_history[i-1]]
    
    # REVISED BROADCAST CLASSIFICATION:
    
    # 1. Successful broadcasts: Thoughtseed changes during eligible windows that lead to state transitions
    successful_broadcasts = []
    
    for ts_change in ts_changes:
        # Look ahead up to 5 timesteps for state transitions
        for offset in range(1, 6):
            check_time = ts_change + offset
            if check_time >= len(time_steps):
                break
                
            # Check if eligible for transition at this time
            if eligible_for_transition[check_time] and check_time in state_transitions:
                successful_broadcasts.append(ts_change)
                break
    
    # 2. Broadcasts that extend current state beyond minimum dwell
    for i in range(len(dominant_ts_history) - 3):
        curr_ts = dominant_ts_history[i]
        if curr_ts in ['breath_focus', 'self_reflection', 'equanimity']:  # Beneficial thoughtseeds
            # Check for 3+ consecutive occurrences during breath_control
            if all(dominant_ts_history[i+j] == curr_ts for j in range(3)):
                curr_state = state_history[i]
                if curr_state == 'breath_control':
                    # If this helped maintain the state beyond min dwell time
                    state_start_idx = max([0] + [st for st in state_transitions if st < i])
                    if i - state_start_idx > min_dwell_times['breath_control']:
                        successful_broadcasts.append(i+1)  # Mark middle of sequence
    
    # 3. All other thoughtseed changes are potential broadcasts
    potential_broadcasts = [i for i in ts_changes if i not in successful_broadcasts]
    
    # Dictionary to track label positions for avoiding overlaps
    label_positions = {}  # Format: {x_position: [y_positions]}
    
    # Draw vertical lines and markers for successful broadcasts
    for sb in successful_broadcasts:
        dominant_ts = dominant_ts_history[sb]
        ts_color = thoughtseed_colors[dominant_ts]
        y_pos = 4 - ts_mapping[dominant_ts]
        
        # Find the next state transition time
        next_trans = next((st for st in state_transitions if st >= sb), None)
        if next_trans and next_trans <= sb + 5:  # Use wider window
            # Draw connecting line between broadcast and state transition
            ax2.plot([sb, sb], [y_pos, 0], linestyle='--', color=ts_color, 
                    linewidth=1.5, alpha=0.7, zorder=12)
            
            # Draw from bottom of state plot to the state transition point
            state_y = reordered_states.index(state_history[next_trans])
            ax1.plot([sb, sb], [3, state_y], linestyle='--', color=ts_color, 
                    linewidth=1.5, alpha=0.7, zorder=12)
        
        # Add subtle highlight background
        ax2.axvspan(sb-0.4, sb+0.4, color=ts_color, alpha=0.1, zorder=4)
        
        # Draw white background circle for contrast
        ax2.scatter(sb, y_pos, s=240, marker='o', facecolors='white', 
                   edgecolors='none', alpha=0.8, zorder=14)
        
        # Draw star marker for successful broadcast
        ax2.scatter(sb, y_pos, s=180, marker='*', facecolors=ts_color,
                edgecolors='white', linewidth=1.5, alpha=1.0, zorder=15)
        
        # Adjust label position to avoid overlaps
        label_offset = 0.6  # Start with a larger offset
        x_rounded = round(sb*2)/2  # Use half-integer rounding for finer grouping
        
        if x_rounded in label_positions:
            while any(abs(y_pos - label_offset - y_pos_used) < 0.5 for y_pos_used in label_positions[x_rounded]):
                label_offset += 0.3  # Increase offset more aggressively
            label_positions[x_rounded].append(y_pos - label_offset)
        else:
            label_positions[x_rounded] = [y_pos - label_offset]
        
        # Use symbols instead of text for less crowding
        ax2.text(sb, y_pos - label_offset, "★", fontsize=14,
                ha='center', va='center', color=ts_color, 
                fontweight='bold', alpha=0.9)
    
    # Draw potential broadcasts with different styling
    for pb in potential_broadcasts:
        dominant_ts = dominant_ts_history[pb]
        ts_color = thoughtseed_colors[dominant_ts]
        y_pos = 4 - ts_mapping[dominant_ts]
        
        # Add subtle background
        ax2.axvspan(pb-0.3, pb+0.3, color='#eeeeee', alpha=0.4, zorder=4)
        
        # Draw background circle
        ax2.scatter(pb, y_pos, s=180, marker='o', facecolors='white',
                   edgecolors='none', alpha=0.8, zorder=14)
        
        # Draw marker for potential broadcast
        ax2.scatter(pb, y_pos, s=120, marker='o', facecolors='none',
                   edgecolors=ts_color, linewidth=2, alpha=1.0, zorder=15)
        
        # Adjust label position
        label_offset = 0.6
        x_rounded = round(pb*2)/2
        
        if x_rounded in label_positions:
            while any(abs(y_pos - label_offset - y_pos_used) < 0.5 for y_pos_used in label_positions[x_rounded]):
                label_offset += 0.3
            label_positions[x_rounded].append(y_pos - label_offset)
        else:
            label_positions[x_rounded] = [y_pos - label_offset]
            
        # Use symbols instead of text
        ax2.text(pb, y_pos - label_offset, "○", fontsize=14, 
                ha='center', va='center', color=ts_color,
                fontweight='bold', alpha=0.9)
    
    ax2.set_yticks(range(len(reordered_thoughtseeds)))
    ax2.set_yticklabels(reordered_thoughtseeds)
    ax2.invert_yaxis()
    ax2.set_title(f'{results["experience_level"].capitalize()}: Thoughtseed Global Workspace Access', 
                 fontsize=13, fontweight='bold')
    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Thoughtseed', fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Legend with improved markers
    legend_items = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=8, label=ts) 
                   for ts, color in thoughtseed_colors.items()]
    
    # Add broadcast markers to legend
    legend_items.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', 
                             markeredgecolor='white', markersize=12, 
                             label='Successful Broadcast'))
    
    legend_items.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                 markeredgecolor='blue', markersize=10, linewidth=2,
                                 label='Potential Broadcast'))
    
    legend = ax2.legend(handles=legend_items, loc='upper right', 
                       bbox_to_anchor=(1.15, 1), framealpha=0.9,
                       fancybox=True, shadow=True, fontsize=10)
    
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#dddddd')
    
    # Updated annotation with dwell time concept
    plt.figtext(0.01, 0.01, 
              "★ Successful Broadcasts: Thoughtseeds that gain access to the global workspace and influence state transitions\n"
              "○ Potential Broadcasts: Thoughtseed activations that occur during ineligible transition periods due to state dwell time constraints\n"
              "Shaded regions in top plot indicate periods when state transitions are possible (after minimum dwell time)",
              fontsize=9, style='italic', ha='left',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='#dddddd', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    plt.savefig(f'./results/plots/simulation_{results["experience_level"]}_global_broadcasting.png', 
               facecolor='white', edgecolor='none', dpi=300, bbox_inches='tight')
    plt.close()
        
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
            plot_thoughtseed_broadcasting(results)
            print(f"Plots for {experience_level} saved to ./results/plots/")
        else:
            print(f"Skipping plot generation for {experience_level} due to missing data")