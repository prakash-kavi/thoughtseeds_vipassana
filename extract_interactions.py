import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

from meditation_config import THOUGHTSEEDS, STATES

# Add ensure_directories function to create necessary directories
def ensure_directories():
    """Create necessary directories for output files"""
    # Create results directory and subdirectories if they don't exist
    os.makedirs('./results/data', exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)
    print("Directories created/verified for output files")

# Configuration dictionary for better parameter management
CONFIG = {
    "causality": {
        "max_lag": 5,
        "significance_threshold": 0.05
    },
    "visualization": {
        "figsize": (10, 8),
        "dpi": 300,
        "cmap": "viridis"  # Red (inhibitory) to Yellow (neutral) to Green (facilitatory)
    },
    "interaction_strengths": {
        "clip_min": -0.7,
        "clip_max": 0.7
    },
    "expertise": {
        "novice": {
            "inhibition_strength": -0.5,
            "facilitation_strength": 0.5,
            "meta_strength": 0.6
        },
        "expert": {
            "inhibition_strength": -0.6,
            "facilitation_strength": 0.7, 
            "meta_strength": 0.7
        }
    }
}

def extract_interaction_matrix(experience_level: str) -> Dict[str, Dict[str, float]]:
    """Main function to extract thoughtseed interaction matrix from training data"""
    print(f"Extracting thoughtseed interaction matrix for {experience_level}...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Load data
    activations_array = load_training_data(experience_level)
    if activations_array is None:
        raise FileNotFoundError(f"Training history for {experience_level} not found. Please ensure the training data is available.")
    
    # Analyze causality using standard methods
    causal_pairs = analyze_granger_causality(activations_array)
    
    # Calculate interaction strengths
    interactions = calculate_interaction_strengths(activations_array, causal_pairs)
    
    # Apply domain knowledge
    supplement_domain_knowledge(interactions, experience_level)
    
    # Save and visualize
    save_and_visualize(interactions, experience_level)
    
    return interactions

def load_training_data(experience_level: str) -> Optional[np.ndarray]:
    """Load training data history and extract activation arrays"""
    try:
        # Fixed path to read from parent's results/data directory
        with open(f"./results/data/learning_{experience_level}_history.pkl", "rb") as f:
            history = pickle.load(f)
        activations_history = history.get('activations_history', [])
        if not activations_history:
            return None
        return np.array(activations_history)
    except FileNotFoundError:
        print("Training history not found.")
        return None

def analyze_granger_causality(activations: np.ndarray) -> List[Tuple[str, str, float]]:
    """Use standard Granger causality test to identify causal relationships"""
    max_lag = CONFIG["causality"]["max_lag"]
    significance = CONFIG["causality"]["significance_threshold"]
    causal_pairs = []
    
    for i, source in enumerate(THOUGHTSEEDS):
        for j, target in enumerate(THOUGHTSEEDS):
            if i == j:  # Skip self-causality
                continue
            
            # Prepare data for Granger test
            data = np.column_stack([activations[:, j], activations[:, i]])
            
            try:
                # Run Granger causality test
                test_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # Extract p-values for each lag
                p_values = [test_results[lag+1][0]['ssr_chi2test'][1] for lag in range(max_lag)]
                min_p = min(p_values)
                
                # If any lag shows significance, consider it causal
                if min_p < significance:
                    # Get the lag with the strongest causality
                    best_lag = p_values.index(min_p) + 1
                    
                    # Standard correlation to determine sign (positive/negative)
                    corr, _ = stats.pearsonr(activations[:, j], activations[:, i])
                    
                    # Add to causal pairs with strength proportional to statistical significance
                    strength = 1.0 - min_p  # Higher strength for lower p-values
                    causal_pairs.append((source, target, strength * np.sign(corr)))
            except:
                # Handle any errors in the statistical test
                continue
    
    return causal_pairs

def calculate_interaction_strengths(
    activations: np.ndarray, causal_pairs: List[Tuple[str, str, float]]
) -> Dict[str, Dict[str, float]]:
    """Calculate interaction strengths based on causality and correlation"""
    interactions = {ts: {"connections": {}} for ts in THOUGHTSEEDS}
    
    # Convert causal pairs to interaction dictionary
    for source, target, strength in causal_pairs:
        # Scale and clip to expected range
        scaled_strength = np.clip(
            strength * 0.7,  # Scale to our target range
            CONFIG["interaction_strengths"]["clip_min"],
            CONFIG["interaction_strengths"]["clip_max"]
        )
        
        interactions[source]["connections"][target] = scaled_strength
    
    return interactions

def get_default_interactions(experience_level: str) -> Dict[str, Dict[str, float]]:
    """Return default interaction matrix based on experience level"""
    if experience_level == "expert":
        return {
            "breath_focus": {"connections": {"pain_discomfort": -0.6, "self_reflection": 0.4, "equanimity": 0.3, "pending_tasks": -0.6}},
            "equanimity": {"connections": {"breath_focus": 0.7, "pain_discomfort": -0.6, "self_reflection": 0.5, "pending_tasks": -0.5}},
            "pain_discomfort": {"connections": {"breath_focus": -0.4, "equanimity": -0.3, "self_reflection": 0.4, "pending_tasks": -0.3}},
            "pending_tasks": {"connections": {"breath_focus": -0.4, "self_reflection": -0.35, "equanimity": -0.3, "pain_discomfort": -0.3}},
            "self_reflection": {"connections": {"breath_focus": 0.4, "equanimity": 0.7, "pain_discomfort": -0.3, "pending_tasks": -0.35}}
        }
    else:  # novice
        return {
            "breath_focus": {"connections": {"pain_discomfort": -0.5, "self_reflection": 0.3, "equanimity": -0.49, "pending_tasks": -0.5}},
            "equanimity": {"connections": {"breath_focus": 0.5, "pain_discomfort": -0.4, "self_reflection": 0.4, "pending_tasks": -0.4}},
            "pain_discomfort": {"connections": {"breath_focus": -0.3, "equanimity": -0.2, "self_reflection": 0.5, "pending_tasks": 0.3}},
            "pending_tasks": {"connections": {"breath_focus": -0.3, "self_reflection": 0.0, "equanimity": -0.2, "pain_discomfort": -0.5}},
            "self_reflection": {"connections": {"breath_focus": 0.3, "equanimity": 0.6, "pain_discomfort": -0.1, "pending_tasks": 0.0}}
        }

def supplement_domain_knowledge(interactions: Dict, experience_level: str) -> None:
    """Add domain-specific knowledge blended with data-driven results"""
    # Get parameters for this experience level
    params = CONFIG["expertise"][experience_level]
    
    # Expert-specific breath focus and equanimity relationship
    if experience_level == "expert":
        if "breath_focus" in interactions and "connections" in interactions["breath_focus"]:
            current_val = interactions["breath_focus"]["connections"].get("equanimity", 0.0)
            domain_val = 0.5  # Increased from 0.3 to ensure stronger connection
            
            # Blend 70% data-driven with 30% domain knowledge
            blend_ratio = 0.5  # Increased from 0.3 to give more weight to domain knowledge
            blended_val = (1-blend_ratio) * current_val + blend_ratio * domain_val
            
            # Apply with a lower threshold to ensure it's included
            if abs(blended_val - current_val) > 0.05:  # Changed from 0.1 to 0.05
                interactions["breath_focus"]["connections"]["equanimity"] = blended_val

def save_and_visualize(interactions: Dict, experience_level: str) -> None:
    """Save interaction matrix and create visualizations"""
    # Save to file
    with open(f"./results/data/extracted_interactions_{experience_level}.pkl", "wb") as f:
        pickle.dump(interactions, f)
        
        # Add JSON export for direct consumption
    import json
    with open(f"./results/data/thoughtseed_interactions_{experience_level}.json", "w") as f:
        json.dump(interactions, f, indent=2)
    
    # Create network visualization
    plot_interaction_network(interactions, experience_level)

def plot_interaction_network(interactions: Dict, experience_level: str) -> None:
    """Visualize the thoughtseed interaction network with improved node placement"""
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Define node categories with consistent ordering
        categories = {
            'breath_focus': 'focus',
            'equanimity': 'emotional_regulation',
            'pain_discomfort': 'distraction',
            'pending_tasks': 'distraction',
            'self_reflection': 'meta-awareness'
        }
        
        # Add nodes with categories
        for ts in THOUGHTSEEDS:
            G.add_node(ts, category=categories.get(ts, 'unknown'))
        
        # Add edges with weights
        for source in interactions:
            connections = interactions[source].get("connections", {})
            for target, weight in connections.items():
                G.add_edge(source, target, weight=weight)
        
        # Create figure
        plt.figure(figsize=CONFIG["visualization"]["figsize"])
        
        # Use a custom layout with strategic node positioning
        # This places nodes in a pentagonal arrangement with related nodes further apart
        pos = {
            'breath_focus': (0.5, 1.0),       # Top center
            'equanimity': (0.9, 0.7),         # Top right
            'self_reflection': (0.1, 0.7),    # Top left
            'pain_discomfort': (0.2, 0.1),    # Bottom left
            'pending_tasks': (0.8, 0.1)       # Bottom right
        }
        
        # Define node colors by category with distinct colors
        category_colors = {
            'focus': '#1f77b4',              # Blue
            'emotional_regulation': '#ff7f0e', # Orange
            'distraction': '#d62728',        # Red
            'meta-awareness': '#2ca02c'      # Green
        }
        
        node_colors = [category_colors[categories[n]] for n in G.nodes]
        
        # Draw nodes with increased size
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.5)
        
        # Draw edges with color based on weight using curved edges for better visibility
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= 0]
        
        # Draw positive (facilitatory) edges in green with curved style
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='green', 
                             width=[abs(G[u][v]['weight'])*3 for u, v in pos_edges], 
                             alpha=0.7, arrows=True, connectionstyle='arc3,rad=0.2')
        
        # Draw negative (inhibitory) edges in red with curved style in opposite direction
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='red',
                             width=[abs(G[u][v]['weight'])*3 for u, v in neg_edges], 
                             alpha=0.7, arrows=True, connectionstyle='arc3,rad=-0.2')
        
        # Draw labels with white background for better visibility
        label_pos = {k: (v[0], v[1]-0.02) for k, v in pos.items()}  # Slightly adjust label positions
        nx.draw_networkx_labels(G, label_pos, font_weight='bold', font_size=10,
                              bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=3))
        
        # Draw edge labels with improved positioning
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                   label_pos=0.3)  # Position labels closer to source node
        
        # Create legend with custom entries
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors['focus'], 
                      markersize=10, label='Focus'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors['emotional_regulation'], 
                      markersize=10, label='Emotional Regulation'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors['distraction'], 
                      markersize=10, label='Distraction'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors['meta-awareness'], 
                      markersize=10, label='Meta-awareness'),
            plt.Line2D([0], [0], color='green', lw=2, label='Facilitatory'),
            plt.Line2D([0], [0], color='red', lw=2, label='Inhibitory')
        ]
        
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 ncol=3, frameon=True, fancybox=True, shadow=True)
        
        plt.title(f"Thoughtseed Interaction Network ({experience_level.title()})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"./results/plots/thoughtseed_network_{experience_level}.png", dpi=CONFIG["visualization"]["dpi"])
        plt.close()
        
        print(f"Network visualization saved as ./results/plots/thoughtseed_network_{experience_level}.png")
    except ImportError:
        print("NetworkX not installed - skipping network visualization")
    except Exception as e:
        print(f"Error creating network visualization: {e}")

if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    
    # Extract and compare both experience levels
    novice_interactions = extract_interaction_matrix("novice")
    expert_interactions = extract_interaction_matrix("expert")
    print_interaction_matrices()
    
    # Use the interaction dictionaries rather than just strings
    plot_interaction_network(novice_interactions, "novice")
    plot_interaction_network(expert_interactions, "expert")
