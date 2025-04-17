# Thoughtseeds Framework: Vipassana Meditation Simulation
This repository contains the Python implementation of the *Thoughtseeds Framework* for simulating thought dynamics in focused-attention Vipassana meditation, as described in:

**"Thoughtseeds Framework: A Hierarchical and Agentic Framework for Investigating Thought Dynamics in Meditative States"**  
Prakash Chandra Kavi, Gorka Zamora-López, Daniel Ari Friedman, Gustavo Patow  
*Entropy*, 2025, DOI: [Pending publication, to be updated]

## Overview

The framework models the four meditative states typically observed in Vipassana meditation (*breath_control*, *mind_wandering*, *meta_awareness*, *redirect_breath*) as emerging from hypothesized thoughtseed dynamics in the brain, as a initial step to modeling thought dynamics. The plots shown in the published paper are available in results/plots. The key papers whose code base helped in developing this codebase are cited below.   

- Sandved-Smith, L.; Hesp, C.; Mattout, J.; Friston, K.; Lutz, A.; Ramstead, M.J.D. Towards a Computational Phenomenology of Mental Action: Modelling Meta-awareness and Attentional Control with Deep Parametric Active Inference. Neurosci. Conscious. 2021, 2021, niab018. https://doi.org/10.1093/nc/niab018.
Code Link - https://colab.research.google.com/drive/1IiMWXRF3tGbVh9Ywm0LuD_Lmurvta04Q?usp=sharing
- Christoff, K.; Irving, Z.C.; Fox, K.C.R.; Spreng, R.N.; Andrews-Hanna, J.R. Mind-wandering as spontaneous thought: A dynamic framework. Nat. Rev. Neurosci. 2016, 17, 718–731. Available online: https://doi.org/10.1038/nrn.2016.113.
- Andrews-Hanna, J.R.; Irving, Z.C.; Fox, K.C.R.; Spreng, R.N.; Christoff, K. 13 The Neuroscience of Spontaneous Thought: An Evolving, Interdisciplinary Field. 2018 The Oxford handbook of spontaneous thought: Mind-wandering, creativity, and dreaming, 143.
- van Vugt, M.; Taatgen, N.; Sackur, J.; Bastian, M. Modeling mind-wandering: A tool to better understand distraction. In Proceedings of the 13th International Conference on Cognitive Modeling, Groningen, The Netherlands, 9–11 April 2015; pp. 237–242. Available online:https://pure.rug.nl/ws/portalfiles/portal/16871364/vanVEtal15.pdf
- Zamora-López, G., Russo, E., Gleiser, P. M., Zhou, C., & Kurths, J. (2011). Characterizing the complexity of brain and mind networks. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 369(1952), 3730-3747.
- Delorme, A.; Brandmeyer, T. Meditation and the Wandering Mind: A Theoretical Framework of Underlying Neurocognitive Mechanisms. Perspect. Psychol. Sci. 2021, 16, 39–66. https://doi.org/10.1177/1745691620917340.
- Hasenkamp, W., & Barsalou, L. W. (2012). Effects of meditation experience on functional connectivity of distributed brain networks. Frontiers in human neuroscience, 6, 38.

Thoughtseeds act as attentional agents, competing to produce state transitions and generating outputs for **Figures 4-8** (weight matrices, activations, state evolution, interaction networks, hierarchical dynamics).

## Repository Structure
This is flat repository structure, and hence I provide high level organization.
- `meditation_config.py`: Configuration settings.
- `learning`: Code for training the model (Section 3 of the paper).
  - `learning_thoughtseeds_revised.py`: Core training logic (`RuleBasedHybridLearner`).
  - `visualize_weight_matrix.py`: Weight matrix visualization (Figure 4 support).
  - `learning_plots.py`: Plots for Figures 5 and 6.
  - `extract_interactions.py`: Thoughtseed interaction network (Figure 7). 
- `simulation`: Code for running simulations (Section 4).
  - `run_simulation.py`: Executes simulation.
  - `meditation_states.py`: State definitions.
  - `metacognition.py`: Meta-awareness dynamics.
  - `param_manager.py`: Parameter handling.
  - `thoughtseed_network.py`: Thoughtseed interactions.
  - `visualize_simulation.py`: Plots for Figures 8.
- `results/`:
  - `data/`: JSON outputs (e.g., `learned_weights_*.json`).
  - `plots/`: PNG figures.
- `extract_interactions.py`: Computes interaction networks (Figure 7 support).

- Running Steps
  1. learning_thoughtseeds_revised.py (Figure 5 and 6) 
  2. visualize_weight_matrix.py (Figure 4)
  3. extract_interactions.py (Figure 7)
  4. run_simulation.py (Figure 8)
