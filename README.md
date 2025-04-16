Thoughtseeds Framework: Vipassana Meditation Simulation

This repository contains the Python implementation of the Thoughtseeds Framework for simulating thought dynamics in focused-attention Vipassana meditation, as described in the paper:

"Thoughtseeds Framework: A Hierarchical and Agentic Framework for Investigating Thought Dynamics in Meditative States"
Prakash Chandra Kavi, Gorka Zamora-López, Daniel Ari Friedman, Gustavo Patow
Entropy, 2025, DOI: [ DOI once available]

Overview

The framework models meditative states (breath_control, mind_wandering, meta_awareness, redirect_breath) using a rule-based hybrid learning approach. Thoughtseeds, acting as attentional agents, compete within a Thoughtseed Network to generate state transitions, producing outputs for Figures 4-8 in the paper (weight matrices, activations, state evolution, interaction networks, hierarchical dynamics).

Repository Structure





learning_thoughtseeds.py: Core simulation script, implementing RuleBasedHybridLearner for training and state dynamics.



visualize_learning/learning_plots.py: Plotting functions for Figures 4-8.



utils/data_handler.py: Utilities for saving JSON outputs (e.g., learned_weights_*.json).



results/data/: Stores simulation outputs (JSONs).



results/plots/: Stores generated plots.
