# Thoughtseeds Framework: Vipassana Meditation Simulation
This repository contains the Python implementation of the *Thoughtseeds Framework* for simulating thought dynamics in focused-attention Vipassana meditation, as described in:

**"Thoughtseeds Framework: A Hierarchical and Agentic Framework for Investigating Thought Dynamics in Meditative States"**  
Prakash Chandra Kavi, Gorka Zamora-López, Daniel Ari Friedman, Gustavo Patow  
Citation: Kavi, P.C.; Zamora-López,G.; Friedman, D.A.; Patow, G.Thoughtseeds: A Hierarchical and Agentic Framework for Investigating Thought Dynamics in Meditative States. 
Entropy 2025, 27, 459. https://doi.org/10.3390/e27050459

## Overview

The framework models the four meditative states typically observed in Vipassana meditation (*breath_control*, *mind_wandering*, *meta_awareness*, *redirect_breath*) as emerging from hypothesized thoughtseed dynamics in the brain, as a initial step to modeling thought dynamics. The plots shown in the published paper are available in results/plots. The key papers whose code base helped in developing this codebase are cited below.   

- Sandved-Smith, L.; Hesp, C.; Mattout, J.; Friston, K.; Lutz, A.; Ramstead, M.J.D. Towards a Computational Phenomenology of Mental Action: Modelling Meta-awareness and Attentional Control with Deep Parametric Active Inference. Neurosci. Conscious. 2021, 2021, niab018. https://doi.org/10.1093/nc/niab018.
Code Link - https://colab.research.google.com/drive/1IiMWXRF3tGbVh9Ywm0LuD_Lmurvta04Q?usp=sharing
- Christoff, K.; Irving, Z.C.; Fox, K.C.R.; Spreng, R.N.; Andrews-Hanna, J.R. Mind-wandering as spontaneous thought: A dynamic framework. Nat. Rev. Neurosci. 2016, 17, 718–731. Available online: https://doi.org/10.1038/nrn.2016.113.
- Andrews-Hanna, J.R.; Irving, Z.C.; Fox, K.C.R.; Spreng, R.N.; Christoff, K. 13 The Neuroscience of Spontaneous Thought: An Evolving, Interdisciplinary Field. 2018 The Oxford handbook of spontaneous thought: Mind-wandering, creativity, and dreaming, 143.
- van Vugt, M.; Taatgen, N.; Sackur, J.; Bastian, M. Modeling mind-wandering: A tool to better understand distraction. In Proceedings of the 13th International Conference on Cognitive Modeling, Groningen, The Netherlands, 9–11 April 2015; pp. 237–242. Available online:https://pure.rug.nl/ws/portalfiles/portal/16871364/vanVEtal15.pdf
- Zamora-López, G., Russo, E., Gleiser, P. M., Zhou, C., & Kurths, J. (2011). Characterizing the complexity of brain and mind networks. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 369(1952), 3730-3747. https://doi.org/10.1098/rsta.2011.0121
- Delorme, A.; Brandmeyer, T. Meditation and the Wandering Mind: A Theoretical Framework of Underlying Neurocognitive Mechanisms. Perspect. Psychol. Sci. 2021, 16, 39–66. https://doi.org/10.1177/1745691620917340.
- Hasenkamp, W., & Barsalou, L. W. (2012). Effects of meditation experience on functional connectivity of distributed brain networks. Frontiers in human neuroscience, 6, 38. 
https://doi.org/10.3389/fnhum.2012.00038

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
  4. run_simulation.py
  5. visualize_simulation.py (Figure 8)

# Self-Organization and Emergence

The Thoughtseeds Framework simulates thought dynamics during Vipassana meditation using principles of **self-organization** and **emergence**. These principles enable complex cognitive behaviors, such as transitions between meditative states (*breath_control*, *mind_wandering*, etc.), to arise from local interactions among attentional agents, or "thoughtseeds" (e.g., `breath_focus`, `pain_discomfort`).

- **Self-Organization**: Local interactions between thoughtseeds evolve dynamically, forming ordered patterns without centralized control. For instance, the `ThoughtseedNetwork.update` method in `thoughtseed_network.py` adjusts activations based on state-specific targets and interaction effects. This fosters competitive dynamics that naturally lead to state transitions.

- **Emergence**: Higher-level meditative states emerge from the interplay of thoughtseed activations and meta-awareness. This is modeled in `metacognition.py` (e.g., the `MetaCognitiveMonitor.update` method), where meta-awareness dynamically modulates thoughtseed competition to produce emergent states.

By connecting **learning mechanisms** (e.g., weight adaptation in `learning_thoughtseeds_revised.py`) with **simulation mechanisms** (e.g., state transitions in `run_simulation.py`), the framework demonstrates how meditative states and transitions arise as emergent phenomena.

---

## Core Concepts

The Thoughtseeds Framework is built upon a hierarchical structure that facilitates **self-organization** and **emergence**. This structure integrates three interconnected levels:

1. **Knowledge Domains**:
   - Represent domain-specific attractors for thoughtseeds.
   - For example, primary and secondary attractors are defined in `learning_thoughtseeds_revised.py` to guide thoughtseed activations during specific meditative states.

2. **Thoughtseed Network**:
   - Models local interactions and activation dynamics.
   - Implemented in `thoughtseed_network.py`, where the `update` method simulates the competitive dynamics among thoughtseeds.

3. **Meta-Cognition**:
   - Regulates thoughtseed competition and state transitions.
   - Implemented in `metacognition.py`, where the `MetaCognitiveMonitor.update` method modulates meta-awareness based on state and habituation.

These levels are interconnected through bidirectional feedback, ensuring that changes in one level influence the others. This hierarchical design enables the framework to model meditative states and transitions realistically and adaptively.
## Learning Mechanisms: Adapting Through Experience

Learning mechanisms adapt the framework’s weights and interactions based on simulated meditation, enabling **self-organization**. These mechanisms are primarily implemented in `learning_thoughtseeds_revised.py` and `extract_interactions.py`.

### Weight Initialization and Updates
- **Implementation**: In `learning_thoughtseeds_revised.py`, the `RuleBasedHybridLearner` class initializes weights in its `__init__` method:
  ```python
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

### Interaction Matrix Extraction
- **Implementation**: In `extract_interactions.py`, the `extract_interaction_matrix` function computes interactions using Granger causality:
  ```python
  def extract_interaction_matrix(experience_level: str) -> Dict[str, Dict[str, float]]:
      activations_array = load_training_data(experience_level)
      causal_pairs = analyze_granger_causality(activations_array)
      interactions = calculate_interaction_strengths(activations_array, causal_pairs)
      supplement_domain_knowledge(interactions, experience_level)
      return interactions
  ```
- **Role**: This identifies causal relationships (e.g., `breath_focus` inhibiting `pending_tasks`), enabling self-organized inhibition patterns that differ by expertise.

### Transition Matrix
- **Implementation**: In `learning_thoughtseeds_revised.py`, the `train` method builds the transition matrix:
  ```python
  self.transition_counts[current_state][next_state] += 1
  ```
  Probabilities are later computed from these counts in the output `state_params`.
- **Role**: Tracks natural state shifts, supporting emergent transitions without hardcoded sequences.

**Adaptive Learning**: These mechanisms learn from simulated data, allowing flexible, experience-dependent self-organization.

---

## Simulation Mechanisms: Dynamic Interactions Driving Emergence

Simulation mechanisms in `thoughtseed_network.py`, `metacognition.py`, and `run_simulation.py` drive the dynamics that produce emergent states and transitions.

### Activation Dynamics
- **Implementation**: In `thoughtseed_network.py`, the `ThoughtseedNetwork.update` method updates activations:
  ```python
  def update(self, meditation_state: str, meta_awareness: float, ...):
      base_targets = self._get_state_activations(meditation_state)
      interaction_effects = {}
      for ts in self.agents:
          interaction_effects[ts] = sum(strength * current_activations[ts] * 0.1 for other_ts, strength in self.interactions[ts].get("connections", {}).items())
      for ts, agent in self.agents.items():
          final_target = base_targets[ts] + interaction_effects.get(ts, 0.0)
          agent.update(final_target, coherent_noise.get(ts, 0.0))
  ```
- **Role**: Activations evolve based on state targets, interactions, and noise, fostering competitive dynamics that lead to emergent attention shifts.

### Meta-Awareness Modulation
- **Implementation**: In `metacognition.py`, the `MetaCognitiveMonitor.update` method adjusts meta-awareness:
  ```python
  def update(self, network: ThoughtseedNetwork, state: str, current_dwell: int, timestep: int) -> float:
      base_awareness = self._calculate_base_awareness(state)
      awareness_with_habituation = base_awareness * (1.0 - self.habituation * 0.3)
      final_awareness = max(0.4, min(1.0, awareness_with_habituatio
n + np.random.normal(0, meta_variance)))
      return final_awareness
  ```
- **Role**: Meta-awareness influences thoughtseed competition, contributing to state emergence (e.g., detecting `mind_wandering`).

### State Transitions
- **Implementation**: In `run_simulation.py`, the `MeditationSimulation.run` method orchestrates transitions via `MeditationStateManager.update`:
  ```python
  self.state_manager.update(self.network, self.metacognition, t)
  ```
  This relies on `thoughtseed_network.py`’s features (e.g., `distraction_level`) and `metacognition.py`’s detection.
- **Role**: Transitions emerge from activation patterns and meta-awareness, not predefined rules.

**Emergence**: The `run` method in `run_simulation.py` iterates these dynamics, producing states naturally.

---

## Why Emergent Phenomena, Not Hard-Coded Rules

- **Dynamic States**: States arise from activation patterns (e.g., high `pain_discomfort` triggers `mind_wandering`).
- **Flexible Transitions**: Shifts depend on runtime dynamics, not fixed logic.
- **Expertise Variation**: Novices and experts differ due to learned parameters, not separate code paths.

---

## Locating the Logic

- **Self-Organization**: `thoughtseed_network.py` (`update`).
- **Learning**: `learning_thoughtseeds_revised.py` (`__init__`, `train`), `extract_interactions.py` (`extract_interaction_matrix`).
- **Emergence**: `run_simulation.py` (`run`), integrating network and metacognition.

---

## Intuition and Design

The framework mimics meditation: thoughtseeds compete, meta-awareness modulates, and states emerge organically. By emphasizing adaptive learning and dynamic interactions, it models cognitive processes realistically, prioritizing emergence over rigid rules.
