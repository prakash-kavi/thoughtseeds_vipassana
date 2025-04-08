import numpy as np
import pickle
import matplotlib.pyplot as plt

import os
import json
import itertools  # For groupby function used in calculating state durations

from learning_plots import plot_results

def ensure_directories():
    """Create necessary directories for output files"""
    # Create results directory and subdirectories if they don't exist
    os.makedirs('./results/data', exist_ok=True)
    os.makedirs('./results/plots', exist_ok=True)
    print("Directories created/verified for output files")

# Define constants locally
MEDITATION_STATE_THOUGHTSEED_ATTRACTORS = {
    "breath_control": {"primary": ["breath_focus"], "secondary": ["equanimity"], "condition": "redirect_breath"},
    "mind_wandering": {"primary": ["pain_discomfort"], "secondary": ["pending_tasks"], "condition": "breath_control"},
    "meta_awareness": {"primary": ["self_reflection"], "secondary": [], "condition": "mind_wandering"},
    "redirect_breath": {"primary": ["equanimity"], "secondary": ["breath_focus"], "condition": "meta_awareness"}
}

STATE_DWELL_TIMES = {
    'novice': {'breath_control': (10, 15), 'mind_wandering': (20, 30), 'meta_awareness': (2, 5), 'redirect_breath': (2, 5)},
    'expert': {'breath_control': (15, 25), 'mind_wandering': (8, 12), 'meta_awareness': (1, 3), 'redirect_breath': (1, 3)}
}

THOUGHTSEED_AGENTS = {
    "breath_focus": {"id": 0, "category": "focus", "intentional_weights": {"novice": 0.8, "expert": 0.95}, "decay_rate": 0.005, "recovery_rate": 0.06},
    "equanimity": {"id": 4, "category": "emotional_regulation", "intentional_weights": {"novice": 0.3, "expert": 0.8}, "decay_rate": 0.008, "recovery_rate": 0.045},
    "pain_discomfort": {"id": 1, "category": "body_sensation", "intentional_weights": {"novice": 0.6, "expert": 0.3}, "decay_rate": 0.02, "recovery_rate": 0.025},
    "pending_tasks": {"id": 2, "category": "distraction", "intentional_weights": {"novice": 0.7, "expert": 0.2}, "decay_rate": 0.015, "recovery_rate": 0.03},
    "self_reflection": {"id": 3, "category": "meta-awareness", "intentional_weights": {"novice": 0.5, "expert": 0.5}, "decay_rate": 0.003, "recovery_rate": 0.015}
}

THOUGHTSEED_INTERACTIONS = {
    "breath_focus": {"connections": {"pain_discomfort": -0.6, "self_reflection": 0.5, "equanimity": 0.5, "pending_tasks": -0.5}},
    "equanimity": {"connections": {"breath_focus": 0.6, "pain_discomfort": -0.6, "self_reflection": 0.5, "pending_tasks": -0.4}},
    "pain_discomfort": {"connections": {"breath_focus": -0.4, "equanimity": -0.3, "self_reflection": 0.5, "pending_tasks": 0.3}},
    "pending_tasks": {"connections": {"breath_focus": -0.3, "self_reflection": 0.3, "equanimity": -0.2, "pain_discomfort": 0.2}},
    "self_reflection": {"connections": {"breath_focus": 0.3, "equanimity": 0.6, "pain_discomfort": -0.2, "pending_tasks": 0.2}}
}

# Noise parameters for biological variability
NOISE_LEVEL = {'novice': 0.08, 'expert': 0.04}

class RuleBasedHybridLearner:
    def __init__(self, experience_level='novice', timesteps_per_cycle=200):
        # Initialize learner with configurable timesteps
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        # Define thoughtseeds and states
        self.thoughtseeds = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
        self.states = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
        self.num_thoughtseeds = len(self.thoughtseeds)
        # Initialize weights with intentional weights, adjusted by attractors
        self.weights = np.zeros((self.num_thoughtseeds, len(self.states)))
        for ts_idx, ts in enumerate(self.thoughtseeds):
            for state_idx, state in enumerate(self.states):
                attrs = MEDITATION_STATE_THOUGHTSEED_ATTRACTORS[state]
                if ts in attrs["primary"]:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.9, 1.1)
                elif ts in attrs["secondary"]:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.7, 0.9)
                else:
                    self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(0.05, 0.2)
        self.weights = np.clip(self.weights, 0.05, 1.0)  # Ensure biological range
        
        # Initialize histories
        self.state_history = []
        self.activations_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        # Map states to indices
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.state_history_over_time = []
        # Dwell times with mean ± 2 SD for variability
        self.min_max_dwell_times = {state: self._calculate_dwell_variability(min_time, max_time)
                                  for state, (min_time, max_time) in STATE_DWELL_TIMES[experience_level].items()}
        # Set noise level for biological variability
        self.noise_level = NOISE_LEVEL[experience_level]
        
        # NEW: Add transition threshold mechanisms
        self.transition_thresholds = {
            'mind_wandering': 0.25,  # Lower from 0.35 to 0.25
            'meta_awareness': 0.30,  # Lower from 0.4 to 0.3
            'return_focus': 0.30     # Lower from 0.4 to 0.3
        }
        
        # NEW: Track transition statistics
        self.transition_counts = {state: {next_state: 0 for next_state in self.states} 
                                for state in self.states}
        
        # NEW: Track activation patterns at transition points
        self.transition_activations = {state: [] for state in self.states}
        
        # NEW: Track natural vs. forced transitions
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # NEW: Track distraction buildup patterns
        self.distraction_buildup_rates = []

    def _calculate_dwell_variability(self, min_time, max_time):
        # Calculate mean ± 2 standard deviations for dwell time variability
        mean_dwell = (min_time + max_time) / 2
        std_dwell = (max_time - min_time) / 6  # Assuming uniform distribution, 2 SD covers ~95% of range
        min_dwell = max(1, int(mean_dwell - 2 * std_dwell))
        max_dwell = int(mean_dwell + 2 * std_dwell)
        return (min_dwell, max_dwell)

    def _get_mean_dwell(self, state):
        # Calculate mean dwell time for a given state and experience level
        min_dwell, max_dwell = STATE_DWELL_TIMES[self.experience_level][state]
        return (min_dwell + max_dwell) / 2

    def get_target_activations(self, state, meta_awareness):
        # Generate target activations based on state, weights, and interactions
        state_idx = self.state_indices[state]
        target_activations = self.weights[:, state_idx].copy()
        attrs = MEDITATION_STATE_THOUGHTSEED_ATTRACTORS[state]

        # Adjust for state-specific dominance with attractors, scaled by meta-awareness
        for ts in self.thoughtseeds:
            ts_idx = self.thoughtseeds.index(ts)
            if ts in attrs["primary"]:
                target_activations[ts_idx] = np.random.uniform(0.45, 0.55) * (1 + meta_awareness * 0.1)
            elif ts in attrs["secondary"]:
                target_activations[ts_idx] = np.random.uniform(0.25, 0.35) * (1 + meta_awareness * 0.1)
            else:
                target_activations[ts_idx] = np.random.uniform(0.05, 0.15) * (1 + meta_awareness * 0.1)

        # Apply thoughtseed interactions for inhibition/facilitation, scaled by mean dwell time
        mean_dwell = self._get_mean_dwell(state)
        dwell_scale = mean_dwell / 30  # Normalize to max mean dwell
        for ts in self.thoughtseeds:
            ts_idx = self.thoughtseeds.index(ts)
            for other_ts, interaction in THOUGHTSEED_INTERACTIONS[ts]["connections"].items():
                other_idx = self.thoughtseeds.index(other_ts)
                adjustment = interaction * target_activations[other_idx] * dwell_scale
                target_activations[ts_idx] += adjustment
                target_activations[ts_idx] = max(0.05, target_activations[ts_idx])  # Ensure no negatives

        # Add noise for biological variability and clip to ensure reasonable bounds
        target_activations += np.random.normal(0, self.noise_level, size=self.num_thoughtseeds)
        target_activations = np.clip(target_activations, 0.05, 1.0)
        return target_activations

    def get_dwell_time(self, state):
        # Generate a random dwell time within mean ± 2 SD for biological variability
        min_dwell, max_dwell = self.min_max_dwell_times[state]
        
        # Get the configured range from STATE_DWELL_TIMES
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Ensure minimal biological plausibility while respecting configured values
        if state in ['meta_awareness', 'redirect_breath']:
            # For brief states: at least 1 timestep, respect configured max
            min_biological = 1
            max_biological = config_max
        else:
            # For longer states: at least 3 timesteps, respect configured max
            min_biological = 3
            max_biological = config_max
        
        # Generate dwell time with proper constraints (respecting both biological needs and configuration)
        return max(min_biological, min(max_biological, int(np.random.uniform(min_dwell, max_dwell))))

    def get_meta_awareness(self, state, activations):
        # Calculate meta-awareness based on state, dominant thoughtseed, and noise
        dominant_ts = self.thoughtseeds[np.argmax(activations)]
        base_awareness = 0.6  # Baseline for minimal self-monitoring
        noise = np.random.normal(0, self.noise_level / 2)  # Reduced noise for stability
        if state == "mind_wandering":
            if dominant_ts in ["pending_tasks", "pain_discomfort"]:
                return max(0.55, base_awareness - 0.05 + noise)
            return max(0.6, base_awareness + noise)
        elif state == "meta_awareness":
            return min(1.0 if self.experience_level == 'expert' else 0.9, base_awareness + 0.4 + noise)
        elif state == "redirect_breath":
            return min(0.85, base_awareness + 0.25 + noise)
        else:  # breath_control
            return min(0.8 if self.experience_level == 'expert' else 0.75, base_awareness + 0.2 + noise)

    def train(self):
        # Train with rule-based dynamics, but allow emergent transitions based on activation patterns
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        current_state_index = 0
        current_state = state_sequence[current_state_index]
        current_dwell = 0
        dwell_limit = self.get_dwell_time(current_state)
        
        # Initialize activations with baseline, adjusted by initial state
        activations = np.full(self.num_thoughtseeds, np.random.uniform(0.05, 0.15))
        activations = self.get_target_activations(current_state, 0.6)
        prev_activations = activations.copy()
        
        # NEW: Track time in focused states for distraction growth
        time_in_focused_state = 0
        
        # NEW: Track statistics for transition patterns
        state_transition_patterns = []
        transition_timestamps = []

        for t in range(self.timesteps):
            meta_awareness = self.get_meta_awareness(current_state, activations)
            target_activations = self.get_target_activations(current_state, meta_awareness)

            # Smooth transition over 3 timesteps for biological plausibility
            if current_dwell < 3:
                alpha = (current_dwell + 1) / 3
                activations = (1 - alpha) * prev_activations + alpha * target_activations * 0.9 + prev_activations * 0.1
            else:
                activations = target_activations * 0.9 + prev_activations * 0.1  # 90% target, 10% current for momentum

            # Apply meta-awareness scaling for state-specific dominance
            if current_state == "mind_wandering" and meta_awareness < 0.6:
                for ts in self.thoughtseeds:
                    i = self.thoughtseeds.index(ts)
                    if ts == "breath_focus":
                        activations[i] *= 0.05  # Strongly suppress breath_focus during low meta-awareness
                    elif ts in ["pain_discomfort", "pending_tasks"]:
                        activations[i] *= 1.2  # Moderate boost for distractions
                    else:
                        activations[i] *= 0.5  # Increase suppression of others for balance
            elif current_state == "meta_awareness" and meta_awareness >= 0.8:
                for ts in self.thoughtseeds:
                    i = self.thoughtseeds.index(ts)
                    if ts == "self_reflection":
                        activations[i] *= 1.5  # Boost self_reflection for higher visibility
                    else:
                        activations[i] *= 0.2  # Strongly suppress others
            elif current_state == "redirect_breath" and meta_awareness >= 0.8:
                for ts in self.thoughtseeds:
                    i = self.thoughtseeds.index(ts)
                    if ts == "equanimity":
                        activations[i] *= 1.5  # Boost equanimity for calm focus
                    elif ts == "breath_focus":
                        activations[i] *= 1.1  # Boost breath_focus slightly for redirection
                    else:
                        activations[i] *= 0.3  # Suppress others
            
            # NEW: Allow distraction growth in focused states (breath_control and redirect_breath)
            if current_state in ["breath_control", "redirect_breath"]:
                time_in_focused_state += 1
                dwell_factor = min(1.0, current_dwell / max(10, dwell_limit))
                
                # Calculate distraction growth based on duration in focused state
                # Experts have lower growth rate than novices
                distraction_scale = 2.5 if self.experience_level == 'novice' else 1.2  # Increased from 1.5/0.7
                distraction_growth = 0.035 * dwell_factor * distraction_scale  # Increased from 0.02
                
                # Record distraction growth rate for statistics
                self.distraction_buildup_rates.append(distraction_growth)
                
                # ADD THIS BLOCK: Boosting factor for occasional strong distractions
                boost_chance = 0.1  # 10% chance of a strong distraction
                boost_factor = 1.0
                if np.random.random() < boost_chance:
                    boost_factor = 3.0  # Triple the growth occasionally for spike distractions
                
                # Apply distraction growth
                for i, ts in enumerate(self.thoughtseeds):
                    if ts in ["pain_discomfort", "pending_tasks"]:
                        # More natural distraction growth
                        activations[i] += distraction_growth* boost_factor
                
                # NEW: Add fatigue to breath focus over time
                for i, ts in enumerate(self.thoughtseeds):
                    if ts == "breath_focus":
                        fatigue_rate = 0.005 if self.experience_level == 'expert' else 0.01
                        fatigue = fatigue_rate * dwell_factor * time_in_focused_state/10
                        activations[i] = max(0.2, activations[i] - fatigue)
            else:
                time_in_focused_state = 0  # Reset counter when not in focused state

            # MODIFIED: Soften caps on thoughtseed activations for more natural patterns
            # Only cap severely excessive activations, not normal ranges
            for i, ts in enumerate(self.thoughtseeds):
                if ts == "pending_tasks" and activations[i] > 0.8:  # Only cap extremely high values
                    activations[i] = 0.8
            
            # Keep activations in biological range
            # Expert-specific learning enhancements
            if self.experience_level == 'expert' and current_state in ["redirect_breath", "meta_awareness"]:
                # Strengthen connection between breath focus and equanimity (facilitates integration)
                bf_idx = self.thoughtseeds.index("breath_focus")
                eq_idx = self.thoughtseeds.index("equanimity")
                
                # If both are active together, slightly boost both (mutual reinforcement)
                if activations[bf_idx] > 0.3 and activations[eq_idx] > 0.3:
                    boost = 0.03 * min(activations[bf_idx], activations[eq_idx])
                    activations[bf_idx] += boost
                    activations[eq_idx] += boost
                    
                # In expert meditators, equanimity helps reduce pain reactivity
                if activations[eq_idx] > 0.4:
                    pd_idx = self.thoughtseeds.index("pain_discomfort")
                    activations[pd_idx] = max(0.05, activations[pd_idx] - 0.02 * activations[eq_idx])
                    
            if self.experience_level == 'expert' and current_state in ["breath_control", "redirect_breath"]:
                bf_idx = self.thoughtseeds.index("breath_focus")
                eq_idx = self.thoughtseeds.index("equanimity")
                
                # Direct influence: Strong breath focus can directly facilitate equanimity 
                # (especially during breath_control state, when breath_focus is naturally high)
                if activations[bf_idx] > 0.4:  # When breath focus is moderately to strongly active
                    # Calculate facilitation effect - stronger with higher breath focus
                    facilitation = 0.08 * activations[bf_idx] 
                    
                    # Apply facilitation with small random variation
                    activations[eq_idx] += facilitation * (1.0 + np.random.uniform(-0.2, 0.2))
                    
                    # Ensure biological range
                    activations[eq_idx] = min(1.0, activations[eq_idx])
            
            activations = np.clip(activations, 0.05, 1.0)

            # Identify dominant thoughtseed
            dominant_ts = self.thoughtseeds[np.argmax(activations)]

            # Track histories for analysis and plotting
            self.state_history.append(current_state)
            self.activations_history.append(activations.copy())
            self.meta_awareness_history.append(meta_awareness)
            self.dominant_ts_history.append(dominant_ts)
            self.state_history_over_time.append(self.state_indices[current_state])

            # Handle state transitions
            if current_dwell >= dwell_limit:
                # Save the activation pattern that led to this transition
                self.transition_activations[current_state].append(activations.copy())
                
                # NEW: Allow natural transitions based on activation patterns (emergent)
                # This creates more varied training data for the simulation
                natural_transition = False
                next_state = None
                
                # Transition probability increases with more training
                natural_prob = 0.4 + min(0.5, t / self.timesteps * 0.6)  # Starts at 40%, grows to 90%  
                
                if np.random.random() < natural_prob:  # Chance for natural transition
                    # Calculate distraction level from pain and pending tasks
                    distraction_level = activations[self.thoughtseeds.index("pain_discomfort")] + \
                                        activations[self.thoughtseeds.index("pending_tasks")]
                    
                    # Check for transition conditions based on activation patterns
                    if current_state in ["breath_control", "redirect_breath"]:
                        # High distraction can lead to mind wandering
                        if distraction_level > self.transition_thresholds['mind_wandering']:
                            next_state = "mind_wandering"
                            natural_transition = True
                            
                        if t % 20 == 0 and current_dwell >= dwell_limit:  # Only log at potential transition points
                            print(f"Timestep {t}: State={current_state}, Distraction={distraction_level:.2f}, " + 
                                f"Self-reflection={activations[self.thoughtseeds.index('self_reflection')]:.2f}, " +
                                f"Breath={activations[self.thoughtseeds.index('breath_focus')]:.2f}, " +
                                f"Threshold={self.transition_thresholds['mind_wandering']:.2f}")
                    
                    elif current_state == "mind_wandering":
                        # High self-reflection can lead to meta-awareness
                        if activations[self.thoughtseeds.index("self_reflection")] > self.transition_thresholds['meta_awareness']:
                            next_state = "meta_awareness"
                            natural_transition = True
                    
                    elif current_state == "meta_awareness":
                        # High breath focus can return to breath control
                        # High equanimity can lead to redirect breath
                        if activations[self.thoughtseeds.index("breath_focus")] > self.transition_thresholds['return_focus']:
                            next_state = "breath_control"
                            natural_transition = True
                        elif activations[self.thoughtseeds.index("equanimity")] > self.transition_thresholds['return_focus']:
                            next_state = "redirect_breath"
                            natural_transition = True
                
                # If no natural transition occurred, follow the fixed sequence
                if not natural_transition:
                    # Follow the sequence as before
                    next_state_index = (current_state_index + 1) % len(state_sequence)
                    next_state = state_sequence[next_state_index]
                    self.forced_transition_count += 1
                else:
                    # Record that we had a natural transition
                    self.natural_transition_count += 1
                    transition_timestamps.append(t)
                    state_transition_patterns.append((current_state, next_state, 
                                                    {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)}))
                
                # Record the transition that occurred
                self.transition_counts[current_state][next_state] += 1
                
                # Update state
                current_state_index = state_sequence.index(next_state)
                current_state = next_state
                current_dwell = 0
                dwell_limit = self.get_dwell_time(current_state)
                
                # Blend activations for smoother transition (70% new target, 30% current)
                new_target = self.get_target_activations(current_state, meta_awareness)
                activations = 0.7 * new_target + 0.3 * activations
            else:
                current_dwell += 1

            prev_activations = activations.copy()
        
        # Add a fallback for minimum natural transitions
        if self.natural_transition_count == 0:
            print("WARNING: No natural transitions occurred. Forcing some natural transitions...")
            # Force at least one natural transition of each type after regular training
            activations = np.full(self.num_thoughtseeds, 0.0)
            
            # Force breath_control to mind_wandering
            activations[self.thoughtseeds.index("pain_discomfort")] = 0.5
            activations[self.thoughtseeds.index("pending_tasks")] = 0.5
            self.transition_activations["breath_control"].append(activations.copy())
            self.transition_counts["breath_control"]["mind_wandering"] += 1
            self.natural_transition_count += 1
            
            # Force mind_wandering to meta_awareness
            activations = np.full(self.num_thoughtseeds, 0.0)
            activations[self.thoughtseeds.index("self_reflection")] = 0.6
            self.transition_activations["mind_wandering"].append(activations.copy())
            self.transition_counts["mind_wandering"]["meta_awareness"] += 1
            
            # Force meta_awareness to breath_control or redirect_breath
            activations = np.full(self.num_thoughtseeds, 0.0)
            activations[self.thoughtseeds.index("breath_focus")] = 0.6
            self.transition_activations["meta_awareness"].append(activations.copy())
            self.transition_counts["meta_awareness"]["breath_control"] += 1

        # Save learned weights
        weight_matrix = self.weights.copy()
        ensure_directories()
        with open(f"./results/data/learned_weights_{self.experience_level}.pkl", "wb") as f:
            pickle.dump(weight_matrix, f)
        
        # NEW: Save transition statistics and thresholds
        transition_stats = {
            'transition_counts': self.transition_counts,
            'transition_thresholds': self.transition_thresholds,
            'natural_transitions': self.natural_transition_count,
            'forced_transitions': self.forced_transition_count,
            'transition_timestamps': transition_timestamps,
            'state_transition_patterns': state_transition_patterns,
            'distraction_buildup_rates': self.distraction_buildup_rates,
            'average_activations_at_transition': {
                state: np.mean(acts, axis=0) if len(acts) > 0 else np.zeros(self.num_thoughtseeds)
                for state, acts in self.transition_activations.items()
            }
        }

        with open(f"./results/data/transition_stats_{self.experience_level}.pkl", "wb") as f:
            pickle.dump(transition_stats, f)

        print(f"Training complete for {self.experience_level}.")
        print(f"  - Weights saved to ./results/data/learned_weights_{self.experience_level}.pkl")
        print(f"  - Transition stats saved to ./results/data/transition_stats_{self.experience_level}.pkl")
        print(f"  - Natural transitions: {self.natural_transition_count}, Forced transitions: {self.forced_transition_count}")

        # Save full learning history for interaction extraction
        history_data = {
            'activations_history': self.activations_history,
            'state_history': self.state_history,
            'meta_awareness_history': self.meta_awareness_history,
            'dominant_ts_history': self.dominant_ts_history,
            'timesteps': self.timesteps
        }

        with open(f"./results/data/learning_{self.experience_level}_history.pkl", "wb") as f:
            pickle.dump(history_data, f)

        print(f"  - Full learning history saved to ./results/data/learning_{self.experience_level}_history.pkl")
        
        print("\nGenerating consumer-ready JSON files...")
    
        # 1. ThoughtseedNetwork parameters
        thoughtseed_params = {
            "interactions": THOUGHTSEED_INTERACTIONS,
            "agent_parameters": {
                ts: {
                    "base_activation": float(np.mean([act[i] for act in self.activations_history])),
                    "responsiveness": float(max(0.5, 1.0 - np.std([act[i] for act in self.activations_history]))),
                    "decay_rate": THOUGHTSEED_AGENTS[ts]["decay_rate"],
                    "recovery_rate": THOUGHTSEED_AGENTS[ts]["recovery_rate"]
                } for i, ts in enumerate(self.thoughtseeds)
            },
            "activation_means_by_state": {
                state: {
                    ts: float(np.mean([
                        self.activations_history[j][i] 
                        for j, s in enumerate(self.state_history) if s == state
                    ])) for i, ts in enumerate(self.thoughtseeds)
                } for state in self.states if any(s == state for s in self.state_history)
            }
        }
        
        with open(f"./results/data/thoughtseed_params_{self.experience_level}.json", "w") as f:
            json.dump(thoughtseed_params, f, indent=2)
        
        # 2. MetaCognition parameters
        meta_params = {
            "transition_thresholds": self.transition_thresholds,
            "meta_awareness_base": 0.8 if self.experience_level == 'expert' else 0.7,
            "meta_awareness_noise": 0.03 if self.experience_level == 'expert' else 0.05,
            "habituation_recovery": 0.5 if self.experience_level == 'expert' else 0.3,
            "average_meta_awareness_by_state": {
                state: float(np.mean([
                    self.meta_awareness_history[j] 
                    for j, s in enumerate(self.state_history) if s == state
                ])) for state in self.states if any(s == state for s in self.state_history)
            }
        }
        
        with open(f"./results/data/metacognition_params_{self.experience_level}.json", "w") as f:
            json.dump(meta_params, f, indent=2)
        
        # 3. MeditationStateManager parameters
        # Convert transition counts to probabilities
        transition_probs = {}
        for source in self.states:
            total = sum(self.transition_counts[source].values())
            if total > 0:
                transition_probs[source] = {
                    target: self.transition_counts[source][target] / total 
                    for target in self.states
                }
            else:
                transition_probs[source] = {target: 0.0 for target in self.states}
        
        state_params = {
            "dwell_times": {
                state: list(STATE_DWELL_TIMES[self.experience_level][state])
                for state in self.states
            },
            "transition_probabilities": transition_probs,
            "state_duration_stats": {
                state: {
                    "mean_duration": float(np.mean([
                        sum(1 for s in g) for k, g in itertools.groupby(self.state_history) 
                        if k == state
                    ]) if any(s == state for s in self.state_history) else 0),
                    "std_duration": float(np.std([
                        sum(1 for s in g) for k, g in itertools.groupby(self.state_history) 
                        if k == state
                    ]) if any(s == state for s in self.state_history) else 0)
                } for state in self.states
            }
        }
        
        with open(f"./results/data/state_params_{self.experience_level}.json", "w") as f:
            json.dump(state_params, f, indent=2)
        
        # 4. Learned weights (as a separate file)
        with open(f"./results/data/learned_weights_{self.experience_level}.json", "w") as f:
            json.dump({
                "weights": weight_matrix.tolist(),
                "thoughtseeds": self.thoughtseeds,
                "states": self.states
            }, f, indent=2)
            
        print(f"  - JSON parameter files saved to ./results/data/ directory")
        
        plot_results(self)
        # plot_transition_stats(self)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Ensure directories exist
    ensure_directories()
    
    # Run for novice
    learner_novice = RuleBasedHybridLearner(experience_level='novice', timesteps_per_cycle=200)
    learner_novice.train()

    # Run for expert
    learner_expert = RuleBasedHybridLearner(experience_level='expert', timesteps_per_cycle=200)
    learner_expert.train()
