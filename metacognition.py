"""
metacognition.py

This module defines the MetaCognitiveMonitor class, which provides meta-awareness tracking 
and mind-wandering detection in the context of a Vipassana simulation. The class evaluates 
meta-cognitive states, adapts based on user experience level, and integrates with the 
ThoughtseedNetwork for real-time monitoring.

Key Features:
- Tracks and adjusts meta-awareness levels based on user states and habituation effects.
- Detects mind-wandering and logs its occurrence with probabilistic mechanisms.
- Handles state transitions and their impact on meta-awareness dynamics.

Classes:
- MetaCognitiveMonitor: Core class for meta-cognitive monitoring during the simulation.

References:
- Metzinger,  2003. Meta awareness paramter. opacity vs transparency
- Sandved-Smith et al., 2021. Meta-cognition as a detection mechanism
"""

import os
import sys
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

from param_manager import SimulationParameterManager
from thoughtseed_network import ThoughtseedNetwork

class MetaCognitiveMonitor:
    """Monitors thoughtseed network activity and provides meta-awareness"""
    
    def __init__(self, experience_level: str, param_manager: SimulationParameterManager):
        """Initialize metacognitive monitoring system"""
        self.experience_level = experience_level
        self.param_manager = param_manager
        self.meta_awareness = 0.7  # Initial value
        self.habituation = 0.0  # Habituation factor (increases with time in state)
        self.meta_awareness_history = []
        self.detection_active = False
        self.detection_chance = 0.0
        
        # Load meta parameters
        self.meta_params = self.param_manager.get_meta_parameters()
        self.transition_stats = self.param_manager.get_transition_stats()
        
    def update(self, network: ThoughtseedNetwork, state: str, 
              current_dwell: int, timestep: int) -> float:
        """Update meta-awareness based on network state and conditions"""
        # Calculate base meta-awareness for current state
        base_awareness = self._calculate_base_awareness(state)
        
        # Calculate habituation effect
        habituation_scale = self._calculate_habituation_scale(state)
        habituation_noise = np.random.normal(0, 0.05)
        self.habituation = min(0.8, (current_dwell / habituation_scale) + habituation_noise)
        
        # Apply habituation effect (reduces awareness over time)
        awareness_with_habituation = base_awareness * (1.0 - self.habituation * 0.3)
        
        # Add state-specific variability
        meta_variance = 0.08 if state == "meta_awareness" else 0.04
        awareness_with_noise = awareness_with_habituation + np.random.normal(0, meta_variance)
        
        # Add sudden insight spikes for novices (rarely)
        if self.experience_level == "novice" and state == "mind_wandering":
            if np.random.random() < 0.03:  # 3% chance
                insight_boost = np.random.uniform(0.2, 0.4)
                print(f"[{timestep}] Sudden insight! Meta-awareness boosted (+{insight_boost:.2f})")
                awareness_with_noise += insight_boost
        
        # Ensure boundaries
        final_awareness = max(0.4, min(1.0, awareness_with_noise))
        
        # Store result
        self.meta_awareness = final_awareness
        self.meta_awareness_history.append(final_awareness)
        
        return final_awareness
    
    def _calculate_base_awareness(self, state: str) -> float:
        """Calculate base meta-awareness level for current state"""
        # State-specific base meta-awareness
        if state == "breath_control":
            return 0.75 if self.experience_level == "expert" else 0.65
        elif state == "mind_wandering":
            # Significantly reduced awareness during mind wandering
            return 0.55
        elif state == "meta_awareness":
            # Peak awareness during meta-awareness state
            return 0.95 if self.experience_level == "expert" else 0.85
        else:  # redirect_breath
            return 0.8 if self.experience_level == "expert" else 0.7
    
    def _calculate_habituation_scale(self, state: str) -> float:
        """Calculate habituation scale factor based on state and experience"""
        if state == "breath_control":
            # Experts habituate more slowly to breath focus
            return 25.0 if self.experience_level == "expert" else 15.0
        elif state == "mind_wandering":
            # Mind wandering habituation happens faster
            return 12.0 if self.experience_level == "expert" else 8.0
        else:
            # Other states have intermediate habituation
            return 18.0 if self.experience_level == "expert" else 12.0
    
    def handle_state_change(self, old_state: str, new_state: str) -> None:
        """Handle state transitions impact on meta-awareness"""
        # Reset habituation on state change
        recovery = self.meta_params["habituation_recovery"]
        self.habituation *= (1.0 - recovery)
        
        # Meta-awareness spike on transition to meta-awareness
        if new_state == "meta_awareness":
            awareness_boost = 0.3 if self.experience_level == "expert" else 0.2
            self.meta_awareness = min(1.0, self.meta_awareness + awareness_boost)
            
        # Meta-awareness drop on transition to mind-wandering
        elif new_state == "mind_wandering":
            awareness_drop = 0.2
            self.meta_awareness = max(0.4, self.meta_awareness - awareness_drop)
    
    def detect_mind_wandering(self, network: ThoughtseedNetwork, 
                             current_state: str, current_dwell: int, 
                             min_dwell: int, max_dwell: int, timestep: int) -> bool:
        """Check if mind wandering is detected"""
        self.detection_active = False
        self.detection_chance = 0.0
        
        if current_state != "mind_wandering":
            return False
            
        # Only check for detection if minimum dwell time has passed
        if current_dwell >= min_dwell:
            # Base detection chance increases with time
            time_factor = (current_dwell - min_dwell) / 5
            
            # Get settings from transition stats or use defaults
            detection_base = 0.10
            detection_growth = 0.03
            if "mind_wandering" in self.transition_stats:
                mw_stats = self.transition_stats["mind_wandering"]
                detection_base = mw_stats.get("detection_base", detection_base)
                detection_growth = mw_stats.get("detection_growth", detection_growth)
            
            base_detection = detection_base + detection_growth * time_factor
            
            # Add factors that affect detection probability
            self_reflection_factor = network.get_feature("self_reflection") * 0.25
            distraction_penalty = network.get_feature("distraction_level") * 0.15
            
            # Calculate final detection probability
            detection_prob = base_detection + self_reflection_factor - distraction_penalty
            detection_prob = min(max(0.05, detection_prob), 0.45)  # Bound between 5-45% per check
            
            # Store detection chance
            self.detection_chance = detection_prob
            
            # Check for detection
            detected = np.random.random() < detection_prob
            
            if detected:
                print(f"[{timestep}] Mind-wandering detected! (p={detection_prob:.2f})")
                self.detection_active = True
                return True
            elif current_dwell % 5 == 0:  # Log periodically
                print(f"[{timestep}] Mind-wandering continues... (p={detection_prob:.2f}, t={current_dwell}/{min_dwell})")
        
        # Force detection at max dwell time with high probability
        if current_dwell >= max_dwell:
            max_dwell_enforce = 0.95 if self.experience_level == "novice" else 0.98
            if np.random.random() < max_dwell_enforce:
                print(f"[{timestep}] Max mind-wandering time reached!")
                self.detection_active = True
                return True
                
        return False

    def get_meta_awareness(self) -> float:
        """Get current meta-awareness level"""
        return self.meta_awareness
    
    def get_meta_awareness_history(self) -> List[float]:
        """Get meta-awareness history"""
        return self.meta_awareness_history.copy()
    
    def is_detection_active(self) -> bool:
        """Check if mind-wandering detection is currently active"""
        return self.detection_active
