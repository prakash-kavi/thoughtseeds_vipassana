"""
learning_thoughtseeds_revised.py

This file implements the Rule-Based Hybrid Learner for simulating thought dynamics in focused-attention 
Vipassana meditation. The code provides a computational framework for modeling meditative states, meta-awareness, 
and state transitions, inspired by concepts from neuroscience and meditation research.

### Key Concepts:
- Thoughtseeds and Meditative States: Defines thoughtseeds (`breath_focus`, `pain_discomfort`, `pending_tasks`, `self_reflection`, `equanimity`) and their interactions 
  with meditative states (`breath_control`, `mind_wandering`, `meta_awareness`, `redirect_breath`) using attractors and state-specific activation patterns.
- Meta-awareness and Attentional Control: Models meta-awareness as a self-monitoring mechanism that regulates state dynamics.
- State Transitions and Neurocognitive Mechanisms: Simulates natural and forced transitions between meditative states 
  based on dynamics such as distraction, fatigue, and activation thresholds.

### Inputs:
- **Configuration**:
  - `STATE_DWELL_TIMES`: Defines state-specific dwell times for novice and expert learners.
  - `THOUGHTSEED_INTERACTIONS`: Specifies inhibitory and facilitative interactions between thoughtseeds.
  - `THOUGHTSEED_AGENTS`: Contains parameters for thoughtseed responsiveness, decay rates, and recovery rates.
  - `MEDITATION_STATE_THOUGHTSEED_ATTRACTORS`: Maps primary and secondary thoughtseeds to states.
  - `NOISE_LEVEL`: Adds variability to simulate biological noise.

- **Simulation Parameters**:
  - `experience_level` (str): Specifies the learner's expertise level (`'novice'` or `'expert'`).
  - `timesteps_per_cycle` (int): Number of timesteps for the simulation (default: 200).

### Outputs:
- **Generated Data** (Saved as JSON files in `./results/data/`):
  1. `transition_stats_{experience_level}.json`: Contains statistics on state transitions (natural vs. forced, timestamps, etc.).
  2. `learning_{experience_level}_history.json`: Records the history of states, meta-awareness, and thoughtseed activations.
  3. `thoughtseed_params_{experience_level}.json`: Includes parameters such as baseline activations and interaction strengths.
  4. `metacognition_params_{experience_level}.json`: Tracks meta-awareness thresholds, noise, and averaged meta-awareness by state.
  5. `state_params_{experience_level}.json`: Contains state-specific parameters like dwell times and transition probabilities.
  6. `learned_weights_{experience_level}.json`: Stores the learned weights for thoughtseeds and states.

### Notes:
- The generated data serves as input for visualization functions provided in `learning_plots.py`.
- Ensure that the required output directories (`./results/data/` and `./results/plots/`) exist before running the simulation.

Relevant Literature:
1. Sandved-Smith et al., 2021. - 3 level hierachical framework
2. Christoff lab papers - Mind wandering dynamics
3. Delorme & Brandmeyer, 2021. Mind wandering dynamics

"""

import numpy as np
import matplotlib.pyplot as plt

import os
import json
import itertools  # For groupby function used in calculating state durations

from visualize_learning.learning_plots import plot_results, plot_side_by_side_transition_matrices
from config.meditation_config import STATE_DWELL_TIMES, THOUGHTSEED_INTERACTIONS, THOUGHTSEED_AGENTS, MEDITATION_STATE_THOUGHTSEED_ATTRACTORS, NOISE_LEVEL
from utils.data_handler import ensure_directories, save_json, load_json

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
        self.weights = self.clip_activations(self.weights)  # Ensure biological range
        
        # Initialize histories
        self.state_history = []
        self.activations_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        # Map states to indices
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.state_history_over_time = []
        # Dwell times with mean ± 2 SD for variability
        self.min_max_dwell_times = {state: self.get_dwell_params(state)[:2] 
                            for state in STATE_DWELL_TIMES[experience_level]}
        # Set noise level for biological variability
        self.noise_level = NOISE_LEVEL[experience_level]
        
        # Transition threshold mechanisms
        self.transition_thresholds = {
            'mind_wandering': 0.25,  # Lower from 0.35 to 0.25
            'meta_awareness': 0.30,  # Lower from 0.4 to 0.3
            'return_focus': 0.30     # Lower from 0.4 to 0.3
        }
        
        # Track transition statistics
        self.transition_counts = {state: {next_state: 0 for next_state in self.states} 
                                for state in self.states}
        
        # Track activation patterns at transition points
        self.transition_activations = {state: [] for state in self.states}
        
        # Track natural vs. forced transitions
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # Track distraction buildup patterns
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
        _, _, mean_dwell = self.get_dwell_params(state)
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
        target_activations = self.clip_activations(target_activations)
        return target_activations
    
    def clip_activations(self, arr):
        # Clip array values to biological range [0.05, 1.0] per paper constraints
        return np.clip(arr, 0.05, 1.0)
        
    def get_dwell_params(self, state):
        # Calculate dwell time params (min, max, mean) from STATE_DWELL_TIMES for given state
        min_time, max_time = STATE_DWELL_TIMES[self.experience_level][state]
        mean_dwell = (min_time + max_time) / 2
        std_dwell = (max_time - min_time) / 6  # 2 SD covers ~95% of uniform range
        min_dwell = max(1, int(mean_dwell - 2 * std_dwell))
        max_dwell = int(mean_dwell + 2 * std_dwell)
        return min_dwell, max_dwell, mean_dwell

    def get_dwell_time(self, state):
        # Generate a random dwell time within mean ± 2 SD, ensuring biological plausibility
        min_dwell, max_dwell, _ = self.get_dwell_params(state)
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Set min/max based on state type for realistic timing
        min_biological = 1 if state in ['meta_awareness', 'redirect_breath'] else 3
        max_biological = config_max
        
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
        
    def apply_expert_boosts(self, activations, state):
        # Apply expert-specific boosts to enhance breath_focus and equanimity integration
        if self.experience_level != 'expert':
            return activations
        bf_idx = self.thoughtseeds.index("breath_focus")
        eq_idx = self.thoughtseeds.index("equanimity")
        pd_idx = self.thoughtseeds.index("pain_discomfort")

        if state in ["redirect_breath", "meta_awareness"]:
            # Mutual reinforcement between breath_focus and equanimity
            if activations[bf_idx] > 0.3 and activations[eq_idx] > 0.3:
                boost = 0.03 * min(activations[bf_idx], activations[eq_idx])
                activations[bf_idx] += boost
                activations[eq_idx] += boost
            # Equanimity reduces pain reactivity
            if activations[eq_idx] > 0.4:
                activations[pd_idx] = max(0.05, activations[pd_idx] - 0.02 * activations[eq_idx])

        if state in ["breath_control", "redirect_breath"]:
            # Breath_focus facilitates equanimity with random variation
            if activations[bf_idx] > 0.4:
                facilitation = 0.08 * activations[bf_idx]
                activations[eq_idx] += facilitation * (1.0 + np.random.uniform(-0.2, 0.2))
                activations[eq_idx] = min(1.0, activations[eq_idx])

        return activations
    
    def log_history(self, state, activations, meta_awareness, dominant_ts):
        # Log all history data for plotting and analysis (Figures 6, 8)
        self.state_history.append(state)
        self.activations_history.append(activations.copy())
        self.meta_awareness_history.append(meta_awareness)
        self.dominant_ts_history.append(dominant_ts)
        self.state_history_over_time.append(self.state_indices[state])
        
    def check_natural_transition(self, activations, current_state, t, natural_prob):
        # Check for natural state transitions based on activation patterns (Section 3.3)
        next_state = None
        natural_transition = False
        
        if np.random.random() < natural_prob:  # Chance increases with training progress
            distraction_level = activations[self.thoughtseeds.index("pain_discomfort")] + \
                                activations[self.thoughtseeds.index("pending_tasks")]
            
            # Transitions from focused states to mind_wandering
            if current_state in ["breath_control", "redirect_breath"]:
                if distraction_level > self.transition_thresholds['mind_wandering']:
                    next_state = "mind_wandering"
                    natural_transition = True
                # Log key metrics every 20 timesteps for debugging
                if t % 20 == 0:
                    print(f"Timestep {t}: State={current_state}, Distraction={distraction_level:.2f}, " +
                        f"Self-reflection={activations[self.thoughtseeds.index('self_reflection')]:.2f}, " +
                        f"Breath={activations[self.thoughtseeds.index('breath_focus')]:.2f}, " +
                        f"Threshold={self.transition_thresholds['mind_wandering']:.2f}")
            
            # Transition from mind_wandering to meta-awareness
            elif current_state == "mind_wandering":
                if activations[self.thoughtseeds.index("self_reflection")] > self.transition_thresholds['meta_awareness']:
                    next_state = "meta_awareness"
                    natural_transition = True
            
            # Transitions from meta_awareness to focused states
            elif current_state == "meta_awareness":
                if activations[self.thoughtseeds.index("breath_focus")] > self.transition_thresholds['return_focus']:
                    next_state = "breath_control"
                    natural_transition = True
                elif activations[self.thoughtseeds.index("equanimity")] > self.transition_thresholds['return_focus']:
                    next_state = "redirect_breath"
                    natural_transition = True
    
        return natural_transition, next_state
    
    def adjust_activations_by_state(self, activations, state, meta_awareness):
        # Adjust activations based on state and meta-awareness for realistic dynamics (Section 3.3)
        if state == "mind_wandering" and meta_awareness < 0.6:
            # Suppress focus, boost distractions during low awareness mind-wandering
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "breath_focus":
                    activations[i] *= 0.05  # Strong suppression
                elif ts in ["pain_discomfort", "pending_tasks"]:
                    activations[i] *= 1.2  # Moderate boost
                else:
                    activations[i] *= 0.5  # General suppression
        elif state == "meta_awareness" and meta_awareness >= 0.8:
            # Boost self-reflection, suppress others during high awareness
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "self_reflection":
                    activations[i] *= 1.5  # Enhance visibility
                else:
                    activations[i] *= 0.2  # Strong suppression
        elif state == "redirect_breath" and meta_awareness >= 0.8:
            # Boost equanimity and breath_focus, suppress others for redirection
            for ts in self.thoughtseeds:
                i = self.thoughtseeds.index(ts)
                if ts == "equanimity":
                    activations[i] *= 1.5  # Calm focus
                elif ts == "breath_focus":
                    activations[i] *= 1.1  # Slight boost
                else:
                    activations[i] *= 0.3  # Suppression
        return activations
    
    def apply_distraction_and_fatigue(self, activations, state, dwell_limit, current_dwell, time_in_focused_state):
        # Apply distraction growth and fatigue in focused states (Section 3.3)
        if state not in ["breath_control", "redirect_breath"]:
            return activations, 0  # Reset time when not in focused state
        
        time_in_focused_state += 1
        dwell_factor = min(1.0, current_dwell / max(10, dwell_limit))
        
        # Calculate distraction growth; higher for novices
        distraction_scale = 2.5 if self.experience_level == 'novice' else 1.2
        distraction_growth = 0.035 * dwell_factor * distraction_scale
        self.distraction_buildup_rates.append(distraction_growth)
        
        # Occasional strong distraction boost (10% chance)
        boost_factor = 3.0 if np.random.random() < 0.1 else 1.0
        
        # Apply distraction to pain_discomfort and pending_tasks
        for i, ts in enumerate(self.thoughtseeds):
            if ts in ["pain_discomfort", "pending_tasks"]:
                activations[i] += distraction_growth * boost_factor
        
        # Apply fatigue to breath_focus based on duration
        for i, ts in enumerate(self.thoughtseeds):
            if ts == "breath_focus":
                fatigue_rate = 0.005 if self.experience_level == 'expert' else 0.01
                fatigue = fatigue_rate * dwell_factor * time_in_focused_state / 10
                activations[i] = max(0.2, activations[i] - fatigue)
        
        return activations, time_in_focused_state
    
    def initialize_training(self):
        # Set up initial state and activations for simulation (Section 3.1)
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        current_state = state_sequence[0]
        current_dwell = 0
        dwell_limit = self.get_dwell_time(current_state)
        activations = np.full(self.num_thoughtseeds, np.random.uniform(0.05, 0.15))
        activations = self.get_target_activations(current_state, 0.6)
        prev_activations = activations.copy()
        time_in_focused_state = 0
        state_transition_patterns = []
        transition_timestamps = []
        return (state_sequence, current_state, current_dwell, dwell_limit, activations, 
                prev_activations, time_in_focused_state, state_transition_patterns, transition_timestamps)
        
    def update_activations(self, activations, prev_activations, current_state, current_dwell, dwell_limit, time_in_focused_state):
        # Update activations with state-specific dynamics (Section 3.3)
        meta_awareness = self.get_meta_awareness(current_state, activations)
        target_activations = self.get_target_activations(current_state, meta_awareness)
        
        # Smooth transition over 3 timesteps for realism
        if current_dwell < 3:
            alpha = (current_dwell + 1) / 3
            activations = (1 - alpha) * prev_activations + alpha * target_activations * 0.9 + prev_activations * 0.1
        else:
            activations = target_activations * 0.9 + prev_activations * 0.1
        
        # Apply state-specific adjustments and distractions
        activations = self.adjust_activations_by_state(activations, current_state, meta_awareness)
        activations, time_in_focused_state = self.apply_distraction_and_fatigue(
            activations, current_state, dwell_limit, current_dwell, time_in_focused_state)
        
        # Soften caps on pending_tasks for natural patterns
        for i, ts in enumerate(self.thoughtseeds):
            if ts == "pending_tasks" and activations[i] > 0.8:
                activations[i] = 0.8
        
        # Apply expert-specific boosts
        activations = self.apply_expert_boosts(activations, current_state)
        activations = self.clip_activations(activations)
        return activations, meta_awareness, time_in_focused_state
    
    def handle_transitions(self, activations, current_state, dwell_limit, current_dwell, t, state_sequence, current_state_index, 
                        state_transition_patterns, transition_timestamps):
        # Manage state transitions when dwell time is exceeded (Section 3.3)
        if current_dwell < dwell_limit:
            return current_state, current_state_index, current_dwell + 1, dwell_limit, activations
        
        self.transition_activations[current_state].append(activations.copy())
        natural_prob = 0.4 + min(0.5, t / self.timesteps * 0.6)
        natural_transition, next_state = self.check_natural_transition(activations, current_state, t, natural_prob)
        
        if not natural_transition:
            next_state_index = (current_state_index + 1) % len(state_sequence)
            next_state = state_sequence[next_state_index]
            self.forced_transition_count += 1
        else:
            self.natural_transition_count += 1
            transition_timestamps.append(t)
            state_transition_patterns.append((current_state, next_state, 
                                            {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)}))
        
        self.transition_counts[current_state][next_state] += 1
        current_state_index = state_sequence.index(next_state)
        current_state = next_state
        current_dwell = 0
        dwell_limit = self.get_dwell_time(current_state)
        
        # Blend activations for smooth transition
        new_target = self.get_target_activations(current_state, self.get_meta_awareness(current_state, activations))
        activations = 0.7 * new_target + 0.3 * activations
        return current_state, current_state_index, current_dwell, dwell_limit, activations
    
    def ensure_minimum_transitions(self, activations):
        # Force minimum natural transitions if none occurred (Section 4.2)
        if self.natural_transition_count == 0:
            print("WARNING: No natural transitions occurred. Forcing some natural transitions...")
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
            
            # Force meta_awareness to breath_control
            activations = np.full(self.num_thoughtseeds, 0.0)
            activations[self.thoughtseeds.index("breath_focus")] = 0.6
            self.transition_activations["meta_awareness"].append(activations.copy())
            self.transition_counts["meta_awareness"]["breath_control"] += 1
        return activations

    def train(self):
        # Train the model with rule-based dynamics and emergent transitions (Section 3)
        (state_sequence, current_state, current_dwell, dwell_limit, activations, 
        prev_activations, time_in_focused_state, state_transition_patterns, 
        transition_timestamps) = self.initialize_training()
        current_state_index = 0  # Initialize index 
        
        for t in range(self.timesteps):
            # Update activations and track meta-awareness
            activations, meta_awareness, time_in_focused_state = self.update_activations(
                activations, prev_activations, current_state, current_dwell, dwell_limit, time_in_focused_state)
            dominant_ts = self.thoughtseeds[np.argmax(activations)]
            self.log_history(current_state, activations, meta_awareness, dominant_ts)
            
            # Handle state transitions
            current_state, current_state_index, current_dwell, dwell_limit, activations = self.handle_transitions(
                activations, current_state, dwell_limit, current_dwell, t, state_sequence, current_state_index,
                state_transition_patterns, transition_timestamps)
            prev_activations = activations.copy()
            
        # Ensure minimum natural transitions
        activations = self.ensure_minimum_transitions(activations)

        # Save training results as JSON files (Section 4)
        weight_matrix = self.weights.copy()
        ensure_directories()  # Ensure output directories exist before saving files
        
        # Save training results as JSON files (Section 4)
        weight_matrix = self.weights.copy()
        ensure_directories()  # Ensure output directories exist

        # Prepare transition statistics
        transition_stats = {
            'transition_counts': self.transition_counts,
            'transition_thresholds': self.transition_thresholds,
            'natural_transitions': self.natural_transition_count,
            'forced_transitions': self.forced_transition_count,
            'transition_timestamps': transition_timestamps,
            'distraction_buildup_rates': self.distraction_buildup_rates,
            'average_activations_at_transition': {
                state: np.mean(acts, axis=0).tolist() if len(acts) > 0 else [0] * len(self.thoughtseeds)
                for state, acts in self.transition_activations.items()
            }
        }

        # Prepare learning history
        history_data = {
            'state_history': self.state_history,
            'meta_awareness_history': [float(ma) for ma in self.meta_awareness_history],
            'dominant_ts_history': self.dominant_ts_history,
            'timesteps': self.timesteps,
            'activations_history': [act.tolist() for act in self.activations_history]
        }

        # Prepare thoughtseed parameters
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
                    ts: float(np.mean([self.activations_history[j][i] for j, s in enumerate(self.state_history) if s == state]))
                    for i, ts in enumerate(self.thoughtseeds)
                } for state in self.states if any(s == state for s in self.state_history)
            }
        }

        # Prepare metacognition parameters
        meta_params = {
            "transition_thresholds": self.transition_thresholds,
            "meta_awareness_base": 0.8 if self.experience_level == 'expert' else 0.7,
            "meta_awareness_noise": 0.03 if self.experience_level == 'expert' else 0.05,
            "habituation_recovery": 0.5 if self.experience_level == 'expert' else 0.3,
            "average_meta_awareness_by_state": {
                state: float(np.mean([self.meta_awareness_history[j] for j, s in enumerate(self.state_history) if s == state]))
                for state in self.states if any(s == state for s in self.state_history)
            }
        }

        # Prepare state parameters
        transition_probs = {
            source: {target: self.transition_counts[source][target] / total if total > 0 else 0.0
                    for target in self.states}
            for source in self.states if (total := sum(self.transition_counts[source].values())) > 0
        }
        state_params = {
            "dwell_times": {state: list(STATE_DWELL_TIMES[self.experience_level][state]) for state in self.states},
            "transition_probabilities": transition_probs or {s: {t: 0.0 for t in self.states} for s in self.states},
            "state_duration_stats": {
                state: {
                    "mean_duration": float(np.mean([sum(1 for s in g) for k, g in itertools.groupby(self.state_history) if k == state])
                                        if any(s == state for s in self.state_history) else 0),
                    "std_duration": float(np.std([sum(1 for s in g) for k, g in itertools.groupby(self.state_history) if k == state])
                                        if any(s == state for s in self.state_history) else 0)
                } for state in self.states
            }
        }

        # List of data to save
        save_list = [
            (transition_stats, f"transition_stats_{self.experience_level}.json"),
            (history_data, f"learning_{self.experience_level}_history.json"),
            (thoughtseed_params, f"thoughtseed_params_{self.experience_level}.json"),
            (meta_params, f"metacognition_params_{self.experience_level}.json"),
            (state_params, f"state_params_{self.experience_level}.json"),
            ({"weights": weight_matrix.tolist(), "thoughtseeds": self.thoughtseeds, "states": self.states},
            f"learned_weights_{self.experience_level}.json")
        ]

        # Save all JSON files and print status
        for data, filename in save_list:
            save_json(data, filename)
        print(f"Training complete for {self.experience_level}. Natural transitions: {self.natural_transition_count}, "
            f"Forced transitions: {self.forced_transition_count}")
        print(f"JSON files saved to ./results/data/: transition_stats, learning_history, thoughtseed_params, "
            f"metacognition_params, state_params, learned_weights")
                
        plot_results(self) 
        
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
