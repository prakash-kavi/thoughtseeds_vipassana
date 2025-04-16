# Thoughtseeds Framework: Vipassana Meditation Simulation

This repository contains the Python implementation of the *Thoughtseeds Framework* for simulating thought dynamics in focused-attention Vipassana meditation, as described in:

**"Thoughtseeds Framework: A Hierarchical and Agentic Framework for Investigating Thought Dynamics in Meditative States"**  
Prakash Chandra Kavi, Gorka Zamora-López, Daniel Ari Friedman, Gustavo Patow  
*Entropy*, 2025, DOI: [Pending publication, to be updated]

## Overview

The framework models meditative states (*breath_control*, *mind_wandering*, *meta_awareness*, *redirect_breath*) using a rule-based hybrid learning approach, inspired by:

- Computational phenomenology for modeling meta-awareness [Sandved-Smith et al., 2021].
- Mind-wandering dynamics from Christoff Lab’s research [Christoff et al., 2016; Andrews-Hanna et al., 2017].
- Computational models of attention shifts [van Vugt et al., 2015].
- Empirical Vipassana meditation studies [Lutz et al., 2015; Hasenkamp & Barsalou, 2012].

Thoughtseeds act as attentional agents, competing to produce state transitions and generating outputs for **Figures 4-8** (weight matrices, activations, state evolution, interaction networks, hierarchical dynamics).

## Repository Structure

- `learning/`: Code for training the model (Section 3 of the paper).
  - `learning_thoughtseeds_revised.py`: Core training logic (`RuleBasedHybridLearner`).
  - `learning_plots.py`: Plots for Figures 4-6.
  - `meditation_config.py`: Configuration settings.
  - `visualize_weight_matrix.py`: Weight matrix visualization (Figure 4 support).
- `simulation/`: Code for running simulations (Section 4).
  - `run_simulation.py`: Executes simulation.
  - `meditation_states.py`: State definitions.
  - `metacognition.py`: Meta-awareness dynamics.
  - `param_manager.py`: Parameter handling.
  - `thoughtseed_network.py`: Thoughtseed interactions.
  - `visualize_simulation.py`: Plots for Figures 7-8.
- `results/`:
  - `data/`: JSON outputs (e.g., `learned_weights_*.json`).
  - `plots/`: PNG figures.
- `extract_interactions.py`: Computes interaction networks (Figure 7 support).

## Requirements

