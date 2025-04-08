import os
import sys
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime

from param_manager import SimulationParameterManager
from thoughtseed_network import ThoughtseedNetwork
from metacognition import MetaCognitiveMonitor
from meditation_states import MeditationStateManager

class MeditationSimulation:
    """Main simulation coordinator"""
    
    def __init__(self, experience_level: str, duration: int = 200):
        """Initialize the meditation simulation"""
        self.experience_level = experience_level
        self.duration = duration
        self.results = {
            'timesteps': duration,
            'experience_level': experience_level,
        }
        
        # Initialize components
        self.param_manager = SimulationParameterManager(experience_level)
        self.network = ThoughtseedNetwork(experience_level)
        self.metacognition = MetaCognitiveMonitor(experience_level, self.param_manager)
        self.state_manager = MeditationStateManager(experience_level, self.param_manager)
        
        # Create output directories
        os.makedirs("./results/data", exist_ok=True)
        os.makedirs("./results/plots", exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """Run the meditation simulation"""
        print(f"\nStarting {self.experience_level} meditation simulation for {self.duration} timesteps...")
        
        # Initialize the network
        self.network.initialize_activations()
        
        # Data collection
        activations_history = []
        meta_awareness_history = []
        state_history = []
        dominant_ts_history = []
        
        # Run simulation
        for t in range(self.duration):
            # Get current state and update meta-awareness
            current_state = self.state_manager.get_current_state()
            meta_awareness = self.metacognition.update(
                self.network, 
                current_state,
                self.state_manager.get_dwell_time(),
                t
            )
            
            # Calculate transition proximity for noise modulation
            transition_proximity = self.state_manager.get_transition_proximity()
            
            # Update network with current context
            self.network.update(
                current_state, 
                meta_awareness, 
                1.0,  # time factor
                self.state_manager.get_focused_time(),
                self.metacognition.is_detection_active(),
                transition_proximity
            )
            
            # Get current activations and dominant thoughtseed
            activations = self.network.get_all_activations()
            dominant_ts = self.network.dominant_seed()
            
            # Update state based on network activity
            self.state_manager.update(self.network, self.metacognition, t)
            
            # Record data
            activations_history.append([activations[ts] for ts in self.param_manager.thoughtseeds])
            meta_awareness_history.append(meta_awareness)
            state_history.append(current_state)
            dominant_ts_history.append(dominant_ts)
            
            # Periodic progress report
            if t % 20 == 0:
                print(f"[{t}] State: {current_state}, Dominant: {dominant_ts}, MA: {meta_awareness:.2f}")
        
        # Store results
        self.results['meta_awareness_history'] = meta_awareness_history
        self.results['activations_history'] = activations_history
        self.results['state_history'] = state_history
        self.results['dominant_ts_history'] = dominant_ts_history
        self.results['thoughtseeds'] = self.param_manager.thoughtseeds
        
        # Save results
        self._save_results()
        
        print(f"\nSimulation complete for {self.experience_level}!")
        return self.results
    
    def _save_results(self) -> None:
        """Save simulation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./results/data/simulation_results_{self.experience_level}_{timestamp}.pkl"
        
        with open(filename, "wb") as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to {filename}")

def main():
    """Main entry point for running simulations"""
    # Create output directories
    os.makedirs("./results/data", exist_ok=True)
    os.makedirs("./results/plots", exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        experience_levels = [sys.argv[1].lower()]
        if experience_levels[0] not in ["novice", "expert"]:
            print("Invalid experience level. Using 'novice' and 'expert'.")
            experience_levels = ["novice", "expert"]
    else:
        experience_levels = ["novice", "expert"]
    
    # Get duration if specified
    duration = 200
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
        except ValueError:
            print(f"Invalid duration. Using default: {duration}")
    
    # Run simulations
    results = {}
    for level in experience_levels:
        print(f"\n{'='*50}")
        print(f"Running {level.upper()} meditation simulation")
        print(f"{'='*50}")
        
        sim = MeditationSimulation(level, duration)
        results[level] = sim.run()
    
    print("\nAll simulations complete!")
    
    # Generate visualizations
    # Note: In a real implementation, we would call visualization.py functions here
    for level, result in results.items():
        print(f"\nVisualizing {level} results...")
        visualize_results(result)


if __name__ == "__main__":
    main()
