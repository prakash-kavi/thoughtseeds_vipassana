"""
thoughtseed_network.py

This file implements the core dynamics of the thoughtseed network, a system of interconnected thoughtseed 
agents representing cognitive processes (e.g., `breath_focus`, `pain_discomfort`, `self_reflection`). 
Each agent's activation evolves over time based on interactions, state-specific parameters, and noise. 
The network tracks dominant thoughtseeds and global workspace competition, which influence meditation 
state transitions.

### Key Responsibilities:
- Initialize thoughtseed agents with parameters provided by the `SimulationParameterManager`.
- Update thoughtseed activations based on state dynamics, interactions, and noise.
- Simulate competitive dynamics within the global workspace and identify dominant thoughtseeds.
- Provide network-level features (e.g., distraction level, meditation quality) for state management.

### Inputs:
- Parameters from `SimulationParameterManager`, such as interaction matrices, state responses, and noise levels.
- State-specific activations and external factors (e.g., meta-awareness, focused time).

### Outputs:
- Current activations of all thoughtseeds.
- Dominant thoughtseed in the global workspace at each timestep.
- Network-level features like distraction level, equanimity, and meditation quality.

### Integration:
- This file interacts closely with `meditation_states.py` for state management and `metacognition.py` for meta-awareness monitoring.
"""
thoughtseed_network.py

This file implements the core dynamics of the thoughtseed network, a system of interconnected thoughtseed 
agents representing cognitive processes (e.g., `breath_focus`, `pain_discomfort`, `self_reflection`). 
Each agent's activation evolves over time based on state-specific parameters, interactions, and noise. 
The network tracks dominant thoughtseeds and global workspace competition, influencing meditation state transitions.

### Key Responsibilities:
- Initialize thoughtseed agents with parameters from `SimulationParameterManager`.
- Update thoughtseed activations dynamically based on interactions, meditation states, and meta-awareness.
- Simulate competitive dynamics within the global workspace and track the dominant thoughtseed.
- Provide network-level features such as distraction level and meditation quality for state management.

### Inputs:
- Parameters from `SimulationParameterManager` (e.g., interaction matrices, state responses, noise levels).
- State-specific activations and external factors (e.g., meta-awareness, focused time).

### Outputs:
- Current activations of all thoughtseeds.
- Dominant thoughtseed in the global workspace at each timestep.
- Network-level features (e.g., distraction level, equanimity, meditation quality).

### Theoretical Foundations:
1. **Mashour, G.A.; Roelfsema, P.; Changeux, J.P.; Dehaene, S.**  "Conscious Processing and the Global Neuronal Workspace Hypothesis."  
   **Neuron, 2020, 105, 776–798.** DOI: [10.1016/j.neuron.2020.01.026](https://doi.org/10.1016/j.neuron.2020.01.026)
   - Provides the theoretical foundation for workspace competition and dominant thoughtseeds.  

2. **Christoff Lab**:  
   - Explores non-linear dynamics of the default mode network (DMN) and its role in transitions between mind-wandering and meta-awareness.  

3. **Van Vugt, M.K.; Christoff, K.; & Schacter, D.L.**  
   - Provides insights into meditation-related dynamics, including the interplay of attentional focus and mind-wandering within neural systems.

4. **Deco, G., et al. (2017)** *"The Dynamics of Intrinsic Ignition: Multistability and Criticality in Cortical Networks."*  
   **Journal of Neuroscience, 37(30), 7603–7618.**  
   - This framework explains how intrinsic ignition events (spontaneous neural activations) propagate through cortical networks, 
     influencing global workspace competition and transitions.  

### Integration:
- Closely interacts with `meditation_states.py` for state transitions and `metacognition.py` for meta-awareness monitoring."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from param_manager import SimulationParameterManager

class ThoughtseedAgent:
    """Individual thoughtseed agent with activation dynamics"""
    
    def __init__(self, name: str, category: str, params: Dict[str, Any]):
        """Initialize a thoughtseed agent"""
        self.name = name
        self.category = category
        self.activation = params.get('initial_activation', 0.2)
        self.base_activation = params.get('base_activation', 0.2)
        self.responsiveness = params.get('responsiveness', 0.7)
        self.recovery_rate = params.get('recovery_rate', 0.05)
        self.decay_rate = params.get('decay_rate', 0.01)
        self.state_responses = params.get('state_responses', {})
        self.activation_history = []
        
    def update(self, target_activation: float, noise: float = 0.0) -> float:
        """Update activation with momentum and noise"""
        # Apply momentum (neural inertia)
        self.activation = (self.responsiveness * target_activation + 
                          (1 - self.responsiveness) * self.activation)
        
        # Add noise
        if noise > 0:
            self.activation += np.random.normal(0, noise)
        
        # Apply biological constraints
        self.activation = np.clip(self.activation, 0.05, 1.0)
        
        # Record history
        self.activation_history.append(self.activation)
        
        return self.activation
        
    def get_activation(self) -> float:
        """Get current activation level"""
        return self.activation
        
    def set_activation(self, value: float) -> None:
        """Set activation to a specific value (for initialization)"""
        self.activation = np.clip(value, 0.05, 1.0)
        
    def get_history(self) -> List[float]:
        """Get activation history"""
        return self.activation_history.copy()


class ThoughtseedNetwork:
    """Network of thoughtseed agents with interaction dynamics"""
    
    def __init__(self, experience_level: str):
        """Initialize network with parameters from learned data"""
        # Load parameters
        self.param_manager = SimulationParameterManager(experience_level)
        self.experience_level = experience_level
        
        # Get thoughtseed list
        self.thoughtseeds = self.param_manager.thoughtseeds
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Get interaction matrix
        self.interactions = self.param_manager.get_interactions()
        
        # State tracking
        self.current_state = "breath_control"
        self.time_in_state = 0
        self.time_in_focused_state = 0
        self.transition_proximity = 0.0
        
        # Network monitoring
        self.dominant_history = []
        self.global_workspace_competition = {}
        
    def _create_agents(self) -> Dict[str, ThoughtseedAgent]:
        """Create thoughtseed agents with learned parameters"""
        agents = {}
        network_params = self.param_manager.get_network_parameters()
        
        # Get learned weights for state responses if available
        weights = self.param_manager.get_weights()
        state_responses = {}
        
        if weights is not None:
            states = self.param_manager.states
            for i, ts in enumerate(self.thoughtseeds):
                state_responses[ts] = {state: weights[i, j] for j, state in enumerate(states)}
        
        # Categories for each thoughtseed based on their nature
        categories = {
            'breath_focus': 'focus',
            'equanimity': 'emotional_regulation',
            'pain_discomfort': 'body_sensation',
            'pending_tasks': 'distraction',
            'self_reflection': 'meta-awareness'
        }
        
        # Create agents with appropriate parameters
        for ts in self.thoughtseeds:
            # Adjust parameters based on thoughtseed type
            if ts in ['breath_focus', 'equanimity']:
                # Focus-related thoughtseeds
                decay_rate = 0.005 if self.experience_level == 'expert' else 0.01
                recovery_rate = 0.06 if self.experience_level == 'expert' else 0.04
                responsiveness = 0.7 if self.experience_level == 'expert' else 0.6
                base_activation = 0.3 if self.experience_level == 'expert' else 0.2
            elif ts in ['pain_discomfort', 'pending_tasks']:
                # Distraction-related thoughtseeds
                decay_rate = 0.015 if self.experience_level == 'expert' else 0.025
                recovery_rate = 0.03 if self.experience_level == 'expert' else 0.05
                responsiveness = 0.8  # Distractions are generally responsive
                base_activation = 0.15 if self.experience_level == 'expert' else 0.25
            else:  # self_reflection
                # Meta-cognitive thoughtseed
                decay_rate = 0.003
                recovery_rate = 0.02 if self.experience_level == 'expert' else 0.015
                responsiveness = 0.75 if self.experience_level == 'expert' else 0.65
                base_activation = 0.2
                
            # Create agent with parameters
            agent_params = {
                'initial_activation': np.random.uniform(0.1, 0.3),
                'base_activation': base_activation,
                'responsiveness': responsiveness,
                'recovery_rate': recovery_rate,
                'decay_rate': decay_rate
            }
            
            # Add state responses if available
            if ts in state_responses:
                agent_params['state_responses'] = state_responses[ts]
            
            agents[ts] = ThoughtseedAgent(ts, categories.get(ts, 'unknown'), agent_params)
            
        return agents
    
    def initialize_activations(self) -> None:
        """Initialize activations for a new simulation"""
        # Set initial activations based on current state
        state_activations = self._get_state_activations("breath_control")
        
        for ts, activation in state_activations.items():
            # Add some random variation
            initial = activation * np.random.uniform(0.8, 1.2)
            self.agents[ts].set_activation(initial)
            
        self.current_state = "breath_control"
        self.time_in_state = 0
        self.time_in_focused_state = 0
    
    def _get_state_activations(self, state: str) -> Dict[str, float]:
        """Get target activations for each thoughtseed based on meditation state"""
        # Default initial activations
        if state == "breath_control":
            return {
                'breath_focus': 0.7 if self.experience_level == 'expert' else 0.6,
                'equanimity': 0.5 if self.experience_level == 'expert' else 0.3,
                'self_reflection': 0.4,
                'pain_discomfort': 0.15,
                'pending_tasks': 0.15
            }
        elif state == "mind_wandering":
            return {
                'breath_focus': 0.15,
                'equanimity': 0.2,
                'self_reflection': 0.3,
                'pain_discomfort': 0.7 if self.experience_level == 'novice' else 0.5,
                'pending_tasks': 0.7 if self.experience_level == 'novice' else 0.5
            }
        elif state == "meta_awareness":
            return {
                'breath_focus': 0.3,
                'equanimity': 0.4,
                'self_reflection': 0.8,
                'pain_discomfort': 0.3,
                'pending_tasks': 0.3
            }
        else:  # redirect_breath
            return {
                'breath_focus': 0.5,
                'equanimity': 0.6 if self.experience_level == 'expert' else 0.4,
                'self_reflection': 0.6,
                'pain_discomfort': 0.2,
                'pending_tasks': 0.2
            }
    
    def update(self, 
               meditation_state: str, 
               meta_awareness: float, 
               time_factor: float = 1.0,
               focused_state_time: int = 0,
               detection_active: bool = False,
               transition_proximity: float = 0.0) -> Dict[str, float]:
        """Update all thoughtseed activations based on current state and interactions"""
        # Update state tracking
        self.current_state = meditation_state
        self.time_in_state += 1
        self.time_in_focused_state = focused_state_time
        self.transition_proximity = transition_proximity
        
        # Get current activations
        current_activations = {ts: agent.get_activation() for ts, agent in self.agents.items()}
        
        # Calculate target activations based on state
        base_targets = self._get_state_activations(meditation_state)
        
        # Apply state-specific modulation
        if meditation_state == "mind_wandering" and meta_awareness < 0.6:
            # Low meta-awareness during mind wandering suppresses focus
            base_targets['breath_focus'] *= 0.7
            base_targets['equanimity'] *= 0.8
        elif meditation_state == "meta_awareness":
            # Higher meta-awareness boosts self-reflection
            boost = 1.0 + (meta_awareness - 0.7) * 0.5
            base_targets['self_reflection'] = min(0.9, base_targets['self_reflection'] * boost)
            
        # Apply detection boost if active
        if detection_active:
            base_targets['self_reflection'] = min(0.9, base_targets['self_reflection'] * 1.3)
        
        # Calculate interactions between thoughtseeds
        interaction_effects = {}
        for ts in self.agents:
            interaction_effects[ts] = 0.0
            if ts in self.interactions:
                connections = self.interactions[ts].get("connections", {})
                for other_ts, strength in connections.items():
                    if other_ts in current_activations:
                        # Scale interaction by current activation of source thoughtseed
                        effect = strength * current_activations[ts] * 0.1
                        interaction_effects[other_ts] = interaction_effects.get(other_ts, 0.0) + effect
        
        # Apply expert-specific dynamics
        if self.experience_level == 'expert':
            if meditation_state in ["breath_control", "redirect_breath"]:
                bf_act = current_activations["breath_focus"]
                eq_act = current_activations["equanimity"]
                
                # Strong breath focus can directly facilitate equanimity
                if bf_act > 0.4:
                    # Calculate facilitation effect based on breath focus strength
                    network_params = self.param_manager.get_network_parameters()
                    facilitation = network_params["equanimity_facilitation"] * bf_act
                    interaction_effects["equanimity"] += facilitation
        
        # Calculate combined targets with interactions
        final_targets = {ts: max(0.05, min(1.0, base_targets[ts] + interaction_effects.get(ts, 0.0))) 
                       for ts in self.agents}

        # Calculate modulated noise based on transition proximity
        base_noise = self.param_manager.get_network_parameters()["noise_level"]
        noise_reduction = self.transition_proximity * 0.7  # Reduce noise by up to 70% near transitions
        noise_level = base_noise * (1.0 - noise_reduction)
        
        # Add temporal coherence to noise (autocorrelated)
        coherent_noise = {}
        for ts in self.agents:
            # Generate noise with temporal correlation
            if hasattr(self, 'previous_noise') and ts in self.previous_noise:
                # Blend previous noise with new noise (30% previous, 70% new)
                coherent_noise[ts] = 0.3 * self.previous_noise[ts] + 0.7 * np.random.normal(0, noise_level)
            else:
                coherent_noise[ts] = np.random.normal(0, noise_level)
        
        # Store for next iteration
        self.previous_noise = coherent_noise
        
        # Update each agent with its target and appropriate noise
        for ts, agent in self.agents.items():
            # Apply state-specific dynamics
            if meditation_state in ["breath_control", "redirect_breath"] and ts in ["pain_discomfort", "pending_tasks"]:
                # Distraction growth in focused states
                time_factor_effect = min(1.0, self.time_in_focused_state / 30)
                growth_factor = 0.015 if self.experience_level == 'expert' else 0.025
                growth = growth_factor * time_factor_effect * time_factor
                final_targets[ts] = min(1.0, final_targets[ts] + growth)
            
            if meditation_state == "mind_wandering" and ts == "self_reflection":
                # Self-reflection growth during mind-wandering
                network_params = self.param_manager.get_network_parameters()
                growth = network_params["self_reflection_growth"] * time_factor
                final_targets[ts] = min(1.0, final_targets[ts] + growth)
            
            # Update with final target and coherent noise
            agent.update(final_targets[ts], coherent_noise.get(ts, 0.0))
        
        # Calculate global workspace competition after update
        self._update_global_workspace()
        
        # Return current activations after update
        return {ts: agent.get_activation() for ts, agent in self.agents.items()}
    
    def _update_global_workspace(self) -> None:
        """Update global workspace based on competitive dynamics"""
        # Get current activations
        current_activations = {ts: agent.get_activation() for ts, agent in self.agents.items()}
        
        # Calculate competition scores (higher = more likely to dominate)
        competition_scores = {}
        
        for ts, activation in current_activations.items():
            # Base score is the activation
            score = activation
            
            # Adjust based on state-specific advantages
            if self.current_state == "breath_control" and ts in ["breath_focus", "equanimity"]:
                score *= 1.2  # Boost focus-related thoughts in breath control
            elif self.current_state == "mind_wandering" and ts in ["pain_discomfort", "pending_tasks"]:
                score *= 1.2  # Boost distractions in mind wandering
            elif self.current_state == "meta_awareness" and ts == "self_reflection":
                score *= 1.3  # Boost self-reflection in meta-awareness
            elif self.current_state == "redirect_breath" and ts in ["breath_focus", "equanimity"]:
                score *= 1.2  # Boost focus and equanimity in redirect breath
            
            competition_scores[ts] = score
        
        # Store competition results
        self.global_workspace_competition = competition_scores
        
        # Determine dominant thoughtseed
        dominant = max(competition_scores, key=competition_scores.get)
        self.dominant_history.append(dominant)
    
    def get_feature(self, feature_name: str) -> float:
        """Get calculated feature based on current thoughtseed activations"""
        # Get current activations
        current_activations = {ts: agent.get_activation() for ts, agent in self.agents.items()}
        
        if feature_name == "distraction_level":
            # Combined activation of distraction thoughtseeds
            return (current_activations["pain_discomfort"] + current_activations["pending_tasks"]) / 2
            
        elif feature_name == "breath_focus":
            # Direct activation of breath focus thoughtseed
            return current_activations["breath_focus"]
            
        elif feature_name == "equanimity":
            # Direct activation of equanimity thoughtseed
            return current_activations["equanimity"]
            
        elif feature_name == "self_reflection":
            # Direct activation of self-reflection thoughtseed
            return current_activations["self_reflection"]
            
        elif feature_name == "meditation_quality":
            # Combined score of positive thoughtseeds
            return (current_activations["breath_focus"] + 
                   current_activations["equanimity"] + 
                   current_activations["self_reflection"] * 0.5) / 2.5
                   
        elif feature_name == "competition_strength":
            # Difference between top two competing thoughtseeds
            if len(self.global_workspace_competition) < 2:
                return 0.0
            sorted_scores = sorted(self.global_workspace_competition.values(), reverse=True)
            return sorted_scores[0] - sorted_scores[1]
        
        # Default fallback
        return 0.0
        
    def dominant_seed(self) -> str:
        """Get the currently dominant thoughtseed in the global workspace"""
        if not self.dominant_history:
            # If history is empty, calculate based on current activations
            current_activations = {ts: agent.get_activation() for ts, agent in self.agents.items()}
            return max(current_activations, key=current_activations.get)
        return self.dominant_history[-1]
    
    def get_all_activations(self) -> Dict[str, float]:
        """Get all current thoughtseed activations"""
        return {ts: agent.get_activation() for ts, agent in self.agents.items()}
    
    def get_activation_history(self) -> Dict[str, List[float]]:
        """Get activation history for all thoughtseeds"""
        return {ts: agent.get_history() for ts, agent in self.agents.items()}

if __name__ == "__main__":
    # Test network initialization and basic update
    for experience_level in ["novice", "expert"]:
        print(f"\nTesting ThoughtseedNetwork for {experience_level}...")
        network = ThoughtseedNetwork(experience_level)
        network.initialize_activations()
        
        # Print initial activations
        print("\nInitial activations:")
        for ts, activation in network.get_all_activations().items():
            print(f"  {ts}: {activation:.3f}")
        
        # Test a few updates in different states
        states = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        for state in states:
            print(f"\nUpdating in {state} state:")
            network.update(state, 0.7, 1.0, 5, False, 0.0)
            activations = network.get_all_activations()
            dominant = network.dominant_seed()
            for ts, activation in activations.items():
                print(f"  {ts}: {activation:.3f}" + (" (DOMINANT)" if ts == dominant else ""))
            print(f"  Distraction level: {network.get_feature('distraction_level'):.3f}")
            print(f"  Meditation quality: {network.get_feature('meditation_quality'):.3f}")
