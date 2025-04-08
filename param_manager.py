import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple

class SimulationParameterManager:
    """Simplified manager for meditation simulation parameters from learned data"""
    
    def __init__(self, experience_level: str):
        """Initialize parameter manager for a specific experience level"""
        self.experience_level = experience_level
        
        # Constants for the simulation
        self.thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
        self.states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
        
        # Define paths for data files
        self.data_dir = "./results/data"
        self.filenames = {
            'interactions': f"{self.data_dir}/extracted_interactions_{experience_level}.pkl",
            'weights': f"{self.data_dir}/learned_weights_{experience_level}.pkl",
            'transitions': f"{self.data_dir}/transition_stats_{experience_level}.pkl",
            'history': f"{self.data_dir}/learning_{experience_level}_history.pkl"
        }
        
        # Ensure output directories exist
        os.makedirs('./results/data', exist_ok=True)
        os.makedirs('./results/plots', exist_ok=True)
        
        # Load configuration data
        self.load_all_config_data()
        
    def load_all_config_data(self) -> bool:
        """Load all configuration data from pickle files and set as attributes"""
        # Track loading status
        all_loaded = True
        
        # Load thoughtseed interactions
        self.THOUGHTSEED_INTERACTIONS = self._load_file('interactions')
        if self.THOUGHTSEED_INTERACTIONS is None:
            all_loaded = False
        
        # Load learned weights
        self.LEARNED_WEIGHTS = self._load_file('weights')
        if self.LEARNED_WEIGHTS is None:
            all_loaded = False
        
        # Load transition statistics
        transitions = self._load_file('transitions')
        if transitions is not None:
            self.TRANSITION_STATS = transitions
            
            # Extract common parameters as direct constants
            self.TRANSITION_THRESHOLDS = transitions.get('transition_thresholds', {})
            self.STATE_DURATION_STATS = transitions.get('state_duration_stats', {})
            self.TRANSITION_ACTIVATIONS = transitions.get('average_activations_at_transition', {})
        else:
            all_loaded = False
        
        # Load history data
        history = self._load_file('history')
        if history is not None:
            self.LEARNING_HISTORY = history
            
            # Calculate useful statistics from history
            if 'activations_history' in history and len(history['activations_history']) > 0:
                activations = np.array(history['activations_history'])
                self.ACTIVATION_MEANS = np.mean(activations, axis=0)
                self.ACTIVATION_STDS = np.std(activations, axis=0)
                
                # Create mapping from thoughtseed index to name
                self.THOUGHTSEED_ACTIVATION_MEANS = {
                    ts: self.ACTIVATION_MEANS[i] for i, ts in enumerate(self.thoughtseeds)
                }
                self.THOUGHTSEED_ACTIVATION_STDS = {
                    ts: self.ACTIVATION_STDS[i] for i, ts in enumerate(self.thoughtseeds)
                }
            
            # Calculate state-specific activation means
            if 'state_history' in history and 'activations_history' in history:
                states = history['state_history']
                activations = history['activations_history']
                if len(states) > 0 and len(activations) > 0:
                    # Group activations by state
                    state_acts = {state: [] for state in self.states}
                    for i in range(min(len(states), len(activations))):
                        if i < len(states) and states[i] in state_acts:
                            state_acts[states[i]].append(activations[i])
                    
                    # Calculate mean activations per state
                    self.STATE_ACTIVATION_MEANS = {}
                    for state, acts in state_acts.items():
                        if acts:
                            mean_acts = np.mean(np.array(acts), axis=0)
                            # Map to thoughtseed names for easier access
                            self.STATE_ACTIVATION_MEANS[state] = {
                                ts: mean_acts[i] for i, ts in enumerate(self.thoughtseeds)
                            }
        
        # Generate state dwell times from stats or use defaults
        self.STATE_DWELL_TIMES = self._get_state_dwell_times()
        
        # Generate agent parameters from activation stats
        self.AGENT_PARAMETERS = self._get_agent_parameters()
        
        # Generate competition modifiers from history
        self.COMPETITION_MODIFIERS = self._get_competition_modifiers()
        
        return all_loaded
    
    def _load_file(self, data_type: str) -> Optional[Any]:
        """Load data from a pickle file"""
        if data_type not in self.filenames:
            print(f"Unknown data type: {data_type}")
            return None
            
        try:
            with open(self.filenames[data_type], 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {data_type} from {self.filenames[data_type]}")
                return data
        except FileNotFoundError:
            print(f"Warning: File not found: {self.filenames[data_type]}")
            return None
        except Exception as e:
            print(f"Error loading {data_type}: {e}")
            return None
    
    def _get_state_dwell_times(self) -> Dict[str, Tuple[int, int]]:
        """Extract state dwell times from statistics"""
        dwell_times = {}
        
        # Try to extract from state duration stats
        if hasattr(self, 'STATE_DURATION_STATS'):
            stats = self.STATE_DURATION_STATS
            
            for state in self.states:
                if state in stats and 'mean_duration' in stats[state] and 'std_duration' in stats[state]:
                    mean = stats[state]['mean_duration']
                    std = stats[state]['std_duration']
                    # Set min/max as mean ± 1.5 std, bounded reasonably
                    dwell_times[state] = (
                        max(1, int(mean - 1.5 * std)),
                        max(5, int(mean + 1.5 * std))
                    )
        
        # Add defaults for missing states
        for state in self.states:
            if state not in dwell_times:
                if state == "breath_control":
                    dwell_times[state] = (15, 25) if self.experience_level == 'expert' else (10, 15)
                elif state == "mind_wandering":
                    dwell_times[state] = (8, 12) if self.experience_level == 'expert' else (20, 30)
                elif state == "meta_awareness":
                    dwell_times[state] = (1, 3) if self.experience_level == 'expert' else (2, 5)
                elif state == "redirect_breath":
                    dwell_times[state] = (1, 3) if self.experience_level == 'expert' else (2, 5)
        
        return dwell_times
    
    def _get_agent_parameters(self) -> Dict[str, Dict[str, float]]:
        """Extract agent parameters from activation statistics"""
        agent_params = {ts: {} for ts in self.thoughtseeds}
        
        # Use activation statistics to determine parameters
        if hasattr(self, 'THOUGHTSEED_ACTIVATION_MEANS') and hasattr(self, 'THOUGHTSEED_ACTIVATION_STDS'):
            for ts in self.thoughtseeds:
                # Base activation from historical mean
                agent_params[ts]['base_activation'] = self.THOUGHTSEED_ACTIVATION_MEANS[ts]
                
                # Responsiveness inverse to historical std
                agent_params[ts]['responsiveness'] = max(0.5, 1.0 - self.THOUGHTSEED_ACTIVATION_STDS[ts])
                
                # Determine decay/recovery based on thoughtseed type
                if ts in ['breath_focus', 'equanimity']:
                    agent_params[ts]['decay_rate'] = 0.005 if self.experience_level == 'expert' else 0.01
                    agent_params[ts]['recovery_rate'] = 0.06 if self.experience_level == 'expert' else 0.04
                elif ts in ['pain_discomfort', 'pending_tasks']:
                    agent_params[ts]['decay_rate'] = 0.015 if self.experience_level == 'expert' else 0.025
                    agent_params[ts]['recovery_rate'] = 0.03 if self.experience_level == 'expert' else 0.05
                else:  # self_reflection
                    agent_params[ts]['decay_rate'] = 0.003
                    agent_params[ts]['recovery_rate'] = 0.02 if self.experience_level == 'expert' else 0.015
        
        return agent_params
    
    def _get_competition_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Calculate competition modifiers from historical data"""
        modifiers = {}
        
        # Try to calculate from history
        if hasattr(self, 'LEARNING_HISTORY'):
            history = self.LEARNING_HISTORY
            if 'state_history' in history and 'dominant_ts_history' in history:
                states = history['state_history']
                dominants = history['dominant_ts_history']
                
                if states and dominants and len(states) == len(dominants):
                    # Count dominant thoughtseeds per state
                    state_dominants = {state: {ts: 0 for ts in self.thoughtseeds} for state in self.states}
                    state_counts = {state: 0 for state in self.states}
                    
                    for i in range(len(states)):
                        state = states[i]
                        dominant = dominants[i]
                        
                        if state in state_counts and dominant in self.thoughtseeds:
                            state_counts[state] += 1
                            state_dominants[state][dominant] += 1
                    
                    # Calculate boost factors based on dominance frequency
                    for state in self.states:
                        if state_counts[state] > 0:
                            modifiers[state] = {}
                            for ts in self.thoughtseeds:
                                dominance = state_dominants[state][ts] / state_counts[state]
                                # Scale from 0.8 to 1.3 based on dominance
                                modifiers[state][ts] = 0.8 + dominance * 0.5
        
        # Add defaults for missing states
        default_modifiers = {
            "breath_control": {
                "breath_focus": 1.2, "equanimity": 1.2, "self_reflection": 1.0,
                "pain_discomfort": 0.8, "pending_tasks": 0.8
            },
            "mind_wandering": {
                "breath_focus": 0.8, "equanimity": 0.8, "self_reflection": 1.0,
                "pain_discomfort": 1.2, "pending_tasks": 1.2
            },
            "meta_awareness": {
                "breath_focus": 0.9, "equanimity": 1.0, "self_reflection": 1.3,
                "pain_discomfort": 0.8, "pending_tasks": 0.8
            },
            "redirect_breath": {
                "breath_focus": 1.2, "equanimity": 1.2, "self_reflection": 1.1,
                "pain_discomfort": 0.8, "pending_tasks": 0.8
            }
        }
        
        for state in self.states:
            if state not in modifiers:
                modifiers[state] = default_modifiers.get(state, {})
        
        return modifiers
    
    def get_network_parameters(self) -> Dict[str, Any]:
        """Get network-level parameters for simulation"""
        # Default values
        network_params = {
            "noise_level": 0.04 if self.experience_level == 'expert' else 0.08,
            "equanimity_facilitation": 0.3 if self.experience_level == 'expert' else 0.15
        }
        
        # Try to load from file if available
        try:
            network_file = f"{self.data_dir}/network_params_{self.experience_level}.json"
            if os.path.exists(network_file):
                import json
                with open(network_file, 'r') as f:
                    loaded_params = json.load(f)
                    network_params.update(loaded_params)
        except Exception as e:
            print(f"Warning: Could not load network parameters from file: {e}")
        
        return network_params
    
    def print_data_summary(self) -> None:
        """Print a summary of the loaded data for verification"""
        print(f"\n===== DATA SUMMARY: {self.experience_level.upper()} =====")
        
        # Print what was loaded
        print("\nLOADED DATA FILES:")
        for data_type, filename in self.filenames.items():
            if hasattr(self, data_type.upper()) and getattr(self, data_type.upper()) is not None:
                print(f"  {data_type}: Loaded successfully")
            else:
                print(f"  {data_type}: Not loaded")
        
        # Print thoughtseed interaction summary
        if hasattr(self, 'THOUGHTSEED_INTERACTIONS'):
            interactions = self.THOUGHTSEED_INTERACTIONS
            print("\nTHOUGHTSEED INTERACTION STATISTICS:")
            for ts in self.thoughtseeds:
                if ts in interactions and 'connections' in interactions[ts]:
                    connections = interactions[ts]['connections']
                    pos_count = sum(1 for v in connections.values() if v > 0)
                    neg_count = sum(1 for v in connections.values() if v < 0)
                    print(f"  {ts}: {len(connections)} connections ({pos_count} positive, {neg_count} negative)")
        
        # Print learned weights shape
        if hasattr(self, 'LEARNED_WEIGHTS') and self.LEARNED_WEIGHTS is not None:
            print(f"\nLEARNED WEIGHTS SHAPE: {self.LEARNED_WEIGHTS.shape}")
        
        # Print transition statistics summary
        if hasattr(self, 'STATE_DURATION_STATS'):
            stats = self.STATE_DURATION_STATS
            print("\nTRANSITION STATISTICS:")
            print("  State Duration Means:")
            for state in self.states:
                if state in stats and 'mean_duration' in stats[state]:
                    print(f"    {state}: {stats[state]['mean_duration']:.2f} timesteps")
        
        # Print agent parameters summary
        if hasattr(self, 'AGENT_PARAMETERS'):
            print("\nAGENT PARAMETERS:")
            for ts, params in self.AGENT_PARAMETERS.items():
                print(f"  {ts}:")
                for param, value in params.items():
                    print(f"    {param}: {value:.4f}")
        
        # Print state target activations
        if hasattr(self, 'STATE_ACTIVATION_MEANS'):
            print("\nSTATE TARGET ACTIVATIONS:")
            for state, activations in self.STATE_ACTIVATION_MEANS.items():
                print(f"  {state}:")
                for ts, value in activations.items():
                    print(f"    {ts}: {value:.2f}")
        
        # Print state dwell times
        if hasattr(self, 'STATE_DWELL_TIMES'):
            print("\nSTATE DWELL TIME RANGES:")
            for state, (min_time, max_time) in self.STATE_DWELL_TIMES.items():
                print(f"  {state}: {min_time}-{max_time} timesteps")
        
        print("\n" + "="*50)


if __name__ == "__main__":
    """Test parameter loading and print data summary"""
    os.makedirs('./results/data', exist_ok=True)
    
    for level in ['novice', 'expert']:
        # Initialize parameter manager for this experience level
        param_manager = SimulationParameterManager(level)
        
        # Print data summary for verification
        param_manager.print_data_summary()