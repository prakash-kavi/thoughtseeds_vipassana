"""
meditation_states.py

This module defines the MeditationStateManager class, which manages the meditation states 
during a Vipassana simulation. It tracks the current state, dwell times, and transitions 
between states based on network activity and user experience level.

Key Features:
- Maintains session factors like mental clarity, distraction susceptibility, and fatigue.
- Detects state transitions using dwell times, distraction levels, and other parameters.
- Integrates with the ThoughtseedNetwork and MetaCognitiveMonitor for decision-making.
- Logs state transitions and provides a history of meditation states.

References:
1. **Christoff Lab**:  
   - Explores non-linear dynamics of the default mode network (DMN) and its role in transitions between mind-wandering and meta-awareness.  
2. **Van Vugt, M.K.; Christoff, K.; & Schacter, D.L.**  
   - Provides insights into meditation-related dynamics, including the interplay of attentional focus and mind-wandering within neural systems.
3. Papers from Delorme, Hasenkamp, Tang and Lutz, especially related mind-wandering and vipassana
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from param_manager import SimulationParameterManager
from thoughtseed_network import ThoughtseedNetwork
from metacognition import MetaCognitiveMonitor

class MeditationStateManager:
    """Manages meditation states based on network activity"""
    
    def __init__(self, experience_level: str, param_manager: SimulationParameterManager):
        """Initialize meditation state manager"""
        self.experience_level = experience_level
        self.param_manager = param_manager
        
        # Current state information
        self.current_state = "breath_control"
        self.current_dwell = 0
        self.time_in_focused_state = 0
        
        # Get dwell time constraints
        self.dwell_times = self.param_manager.get_state_dwell_times()
        
        # Transition tracking
        self.last_transition_time = 0
        self.transition_log = []
        self.state_history = []
        
        # Initialize session factors
        self.session_factors = {
            "mental_clarity": round(np.random.uniform(0.7, 1.0) if experience_level == "expert" else np.random.uniform(0.4, 0.8), 2),
            "distraction_susceptibility": round(np.random.uniform(0.3, 0.6) if experience_level == "expert" else np.random.uniform(0.5, 0.9), 2),
            "fatigue": round(np.random.uniform(0, 0.3), 2),
            "daily_variance": np.random.uniform(0.85, 1.15)
        }
        print(f"Session factors: {self.session_factors}")
    
    def update(self, network: ThoughtseedNetwork, metacognition: MetaCognitiveMonitor, timestep: int) -> bool:
        """Update meditation state based on network activations"""
        # First check for mind-wandering detection
        detection = metacognition.detect_mind_wandering(
            network, 
            self.current_state, 
            self.current_dwell, 
            self.dwell_times[self.current_state][0], 
            self.dwell_times[self.current_state][1],
            timestep
        )
        
        # If detection occurred, transition to meta-awareness
        if detection and self.current_state == "mind_wandering":
            self.transition_to("meta_awareness", "Mind-wandering detected", timestep)
            metacognition.handle_state_change(self.current_state, "meta_awareness")
            return True
        
        # Handle state-specific transitions
        transition_occurred = self._check_state_transitions(network, timestep)
        
        # Update state dwell time if no transition occurred
        if not transition_occurred:
            self.current_dwell += 1
            
            # Update focused state time for appropriate states
            if self.current_state in ["breath_control", "redirect_breath"]:
                self.time_in_focused_state += 1
            
        # Record state history
        self.state_history.append(self.current_state)
        
        return transition_occurred
    
    def _check_state_transitions(self, network: ThoughtseedNetwork, timestep: int) -> bool:
        """Check for state transitions based on network state"""
        # Calculate transition readiness
        transition_readiness = self._calculate_transition_readiness()
        
        # Check state-specific transitions
        if self.current_state == "breath_control":
            return self._check_breath_control_transitions(network, transition_readiness, timestep)
        elif self.current_state == "meta_awareness":
            return self._check_meta_awareness_transitions(network, transition_readiness, timestep)
        elif self.current_state == "redirect_breath":
            return self._check_redirect_breath_transitions(network, transition_readiness, timestep)
        
        # Mind wandering transitions are handled by metacognition.detect_mind_wandering
        return False
    
    def _calculate_transition_readiness(self) -> float:
        """Calculate how ready the state is for transition based on dwell time"""
        if self.current_dwell < self.dwell_times[self.current_state][0]:
            return 0.0
        
        # Gradually increase transition probability between min and max dwell time
        transition_range = self.dwell_times[self.current_state][1] - self.dwell_times[self.current_state][0]
        if transition_range > 0:
            progress = (self.current_dwell - self.dwell_times[self.current_state][0]) / transition_range
            return min(1.0, progress * 1.5)  # Scales from 0 to 1.5
        return 0.0
    
    def _check_breath_control_transitions(self, network: ThoughtseedNetwork, 
                                        transition_readiness: float, timestep: int) -> bool:
        """Check for transitions from breath control to mind wandering"""
        # Get distraction level
        distraction_level = network.get_feature("distraction_level")
        
        # Force transition if max dwell time is reached
        if self.current_dwell >= self.dwell_times[self.current_state][1]:
            self.transition_to("mind_wandering", "Max breath control time reached", timestep)
            return True
            
        # Natural transition based on distraction level
        distraction_threshold = 0.45 if self.experience_level == "novice" else 0.55
        if distraction_level > distraction_threshold:
            # Higher distraction increases chance of transition
            variance_factor = 0.4 if self.experience_level == "novice" else 0.8
            transition_prob = distraction_level * transition_readiness * variance_factor
            
            if np.random.random() < transition_prob:
                self.transition_to("mind_wandering", "High distraction", timestep)
                return True
                
        # Spontaneous mind-wandering (more likely for novices, increases over time)
        spontaneous_chance = 0.03 if self.experience_level == "novice" else 0.015
        time_factor = self.time_in_focused_state / 15
        spontaneous_prob = spontaneous_chance * time_factor * self.session_factors["distraction_susceptibility"]
        
        if np.random.random() < spontaneous_prob:
            self.transition_to("mind_wandering", "Attention shift", timestep)
            return True
            
        return False
    
    def _check_meta_awareness_transitions(self, network: ThoughtseedNetwork, 
                                        transition_readiness: float, timestep: int) -> bool:
        """Check for transitions from meta awareness to redirect breath"""
        # Meta-awareness always transitions to redirect_breath after sufficient time
        min_transition_prob = 0.3 if self.experience_level == "novice" else 0.4
        variability = 0.5 if self.experience_level == "novice" else 0.4
        
        # Calculate transition probability
        base_prob = min_transition_prob + (1 - min_transition_prob) * transition_readiness
        transition_prob = base_prob * variability * self.session_factors["daily_variance"]
        
        # Force transition at max dwell time or randomly based on probability
        if (self.current_dwell >= self.dwell_times[self.current_state][1] or 
            np.random.random() < transition_prob):
            self.transition_to("redirect_breath", "Meta-awareness complete", timestep)
            return True
            
        return False
    
    def _check_redirect_breath_transitions(self, network: ThoughtseedNetwork, 
                                         transition_readiness: float, timestep: int) -> bool:
        """Check for transitions from redirect breath to breath control"""
        # Get breath focus level
        breath_focus = network.get_feature("breath_focus")
        
        # Success threshold based on experience
        success_threshold = 0.4 if self.experience_level == "novice" else 0.45
        success_prob_scale = 0.8 if self.experience_level == "novice" else 1.0
        
        # Calculate success probability
        success_prob = breath_focus * transition_readiness * success_prob_scale
        
        # Force transition at max dwell time or when breath focus is restored
        if self.current_dwell >= self.dwell_times[self.current_state][1]:
            self.transition_to("breath_control", "Max redirection time reached", timestep)
            return True
        elif breath_focus > success_threshold and np.random.random() < success_prob:
            self.transition_to("breath_control", "Breath focus restored", timestep)
            return True
            
        return False
    
    def transition_to(self, new_state: str, reason: str, timestep: int) -> None:
        """Transition to a new meditation state"""
        old_state = self.current_state
        print(f"[{timestep}] {reason} - transitioning to {new_state}")
        
        # Reset counters
        self.last_transition_time = timestep
        self.current_dwell = 0
        
        if new_state != old_state:
            # Reset focused state time except for meta-awareness
            if new_state != "meta_awareness":
                self.time_in_focused_state = 0
            
            # Log the transition
            self.transition_log.append({
                'from': old_state,
                'to': new_state,
                'time': timestep,
                'reason': reason
            })
            
        # Set new state
        self.current_state = new_state
    
    def get_current_state(self) -> str:
        """Get the current meditation state"""
        return self.current_state
    
    def get_dwell_time(self) -> int:
        """Get current dwell time in the current state"""
        return self.current_dwell
    
    def get_focused_time(self) -> int:
        """Get time spent in focused state"""
        return self.time_in_focused_state
    
    def get_state_history(self) -> List[str]:
        """Get history of meditation states"""
        return self.state_history.copy()
    
    def get_transition_proximity(self) -> float:
        """Calculate proximity to next transition based on dwell times"""
        min_dwell = self.dwell_times[self.current_state][0]
        max_dwell = self.dwell_times[self.current_state][1]
        
        # Before min dwell time
        if self.current_dwell < min_dwell:
            return 0.0
            
        # At or past max dwell time
        if self.current_dwell >= max_dwell:
            return 1.0
            
        # In between min and max
        progress = (self.current_dwell - min_dwell) / (max_dwell - min_dwell)
        return min(1.0, progress * 1.2)  # Scale slightly to anticipate transition
