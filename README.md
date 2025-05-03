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

# Self-Organization and Emergence in Thoughtseeds Framework

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

---

## Learning Mechanisms: Adapting Through Experience

The Thoughtseeds Framework employs adaptive learning mechanisms to dynamically adjust weights, interaction patterns, and transition probabilities based on simulated meditation experiences. These mechanisms are inspired by the way meditative practitioners refine their cognitive processes through repeated practice and feedback. The framework’s learning algorithms enable **self-organization**, allowing the system to model the nuanced progression from novice to expert meditators.

### Key Features of Learning Mechanisms

1. **Dynamic Weight Adaptation**:  
   **Primary Attractors**: Thoughtseeds most relevant to a meditative state are given higher initial weights.  
   **Secondary Attractors**: Thoughtseeds indirectly related to a meditative state are assigned moderate weights.  
   **Stochastic Adjustments**: Random perturbations within defined ranges simulate individual variability and learning uncertainty.  

   - The framework adapts the interaction weights between thoughtseeds and meditative states based on simulated experiences.
   - Weights are initialized and updated using a hybrid rule-based and stochastic approach, ensuring both structure and flexibility.
   - Implementation is found in `learning_thoughtseeds_revised.py`, specifically within the `RuleBasedHybridLearner` class.

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
   
2. **Interaction Matrix Extraction**  
   - **Causal Relationships**: The interaction matrix identifies causal relationships between thoughtseeds, revealing how their activations influence each other during meditation.  
   - **Granger Causality**: Relationships are computed using Granger causality, which analyzes temporal dependencies in activation dynamics.  
   - **Adaptive Refinement**: Interactions are supplemented with domain knowledge to reflect expertise-dependent dynamics. For example, `breath_focus` may inhibit `pending_tasks` more effectively for experts.  
   - **Implementation**: Found in `extract_interactions.py`, the `extract_interaction_matrix` function computes these relationships and interaction strengths:  
     ```python
     def extract_interaction_matrix(experience_level: str) -> Dict[str, Dict[str, float]]:
         activations_array = load_training_data(experience_level)
         causal_pairs = analyze_granger_causality(activations_array)
         interactions = calculate_interaction_strengths(activations_array, causal_pairs)
         supplement_domain_knowledge(interactions, experience_level)
         return interactions
     ```  
   - **Role**: Enables self-organized inhibition and facilitation patterns, adapting over time to reflect meditative expertise.

3. **Transition Matrix Construction**  
    - **Tracking Transitions**: Transition matrices monitor the frequency of state-to-state shifts, capturing emergent dynamics instead of relying on predefined sequences.
    - **State Probabilities**: Transition probabilities are derived from accumulated counts of observed shifts during training.
    - **Implementation**: Found in `learning_thoughtseeds_revised.py`, the `train` method updates the transition matrix incrementally:
      ```python
      # Increment transition counts for observed state changes
      self.transition_counts[current_state][next_state] += 1
      
      # Compute transition probabilities from counts
      for state in self.transition_counts:
          total_transitions = sum(self.transition_counts[state].values())
          self.transition_probs[state] = {
              next_state: count / total_transitions
              for next_state, count in self.transition_counts[state].items()
          }
      ```
     - **Role**: Supports emergent transitions by modeling natural shifts in meditative states, such as moving from `mind_wandering` to `meta_awareness` without hardcoded rules.

4. **Expertise-Dependent Learning**  
   - **Progression Modeling**:  
     The framework adjusts learning parameters to simulate progression from novice to expert meditators.  
   - **Behavioral Differences**:  
     Novices exhibit weaker inhibition of distractions such as `pending_tasks`, while experts demonstrate stronger and more consistent suppression.  
   - **Implementation**:  
     Expertise levels are modeled through parameter scaling and experience-based adjustments in `learning_thoughtseeds_revised.py`. For example:  
     ```python
     if experience_level == "novice":
         inhibition_strength = BASE_INHIBITION * 0.5
     elif experience_level == "expert":
         inhibition_strength = BASE_INHIBITION * 1.5
     ```
   - **Role**:  
     Captures the progressive refinement of cognitive control and attentional focus, enabling realistic simulations of meditative expertise.
     
5. **Stochasticity and Variability**  
   - **Simulating Real-World Variability**:  
     Controlled randomness is introduced to replicate individual differences and real-world inconsistencies during meditation.  
   - **Exploration vs. Exploitation**:  
     Noise facilitates the exploration of novel state transitions while reinforcing successful meditative patterns. This balance prevents the framework from converging prematurely on suboptimal solutions.  
   - **Implementation**:  
     Found in `learning_thoughtseeds_revised.py`, stochastic adjustments are incorporated into weight adaptation, ensuring diversity in learning outcomes. For example:  
     ```python
     self.weights[ts_idx, state_idx] = THOUGHTSEED_AGENTS[ts]["intentional_weights"][experience_level] * np.random.uniform(lower_bound, upper_bound)
     ```
   - **Role**:  
     Promotes robustness and adaptability in learning by simulating variability in meditation experiences.

6. **Integration of Domain Knowledge**  
   - **Enhancing Learning**:  
     Expert insights into meditation practices and cognitive dynamics are embedded to refine the framework's learning mechanisms.  
   - **Validated Interactions**:  
     Causal relationships between thoughtseeds are supplemented with theoretical and empirical knowledge to ensure alignment with meditative principles.  
   - **Implementation**:  
     Found in `extract_interactions.py`, the `supplement_domain_knowledge` function enriches interaction matrices with domain expertise:  
     ```python
     def supplement_domain_knowledge(interactions: Dict[str, Dict[str, float]], experience_level: str):
         for ts, connections in interactions.items():
             if ts in DOMAIN_KNOWLEDGE_PRIORITIES:
                 prioritized_weights = DOMAIN_KNOWLEDGE_PRIORITIES[ts][experience_level]
                 for related_ts, weight in prioritized_weights.items():
                     connections[related_ts] += weight
     ```
   - **Role**:  
     Ensures that learned patterns remain grounded in established scientific and meditative frameworks, enabling the framework to reflect realistic dynamics.

     ### Connections Between Features
   The Thoughtseeds Framework integrates multiple learning mechanisms to simulate meditation as a dynamic, emergent process. Explicit connections between these mechanisms enhance the framework’s adaptability and realism:
   
   1. **Interaction and Transition Matrices**:  
      - The **interaction matrix**, computed in `extract_interactions.py`, identifies causal relationships between thoughtseeds. These relationships directly influence how states transition in the **transition matrix**, found in `learning_thoughtseeds_revised.py`.  
      - For example, stronger inhibition of `pending_tasks` by `breath_focus` reduces the likelihood of transitioning into `mind_wandering`, thereby modifying the probabilities in the transition matrix.
      - Code Example:  
        ```python
        def adjust_transition_matrix(interactions: Dict[str, Dict[str, float]]):
            for state, transitions in self.transition_probs.items():
                for target_state in transitions:
                    interaction_effect = calculate_interaction_effect(state, target_state, interactions)
                    transitions[target_state] *= interaction_effect
        ```

   2. **Feedback-Driven Adaptation and Expertise-Dependent Learning**:  
      - Feedback loops refine weights and transitions, which are further scaled by expertise levels to simulate progression from novice to expert meditators.  
      - For instance, a novice may receive feedback to strengthen `meta_awareness` transitions, while an expert focuses on sustaining `breath_control` under distraction.  

   3. **Dynamic Weight Adaptation and Stochasticity**:  
      - Random perturbations in weight updates (stochasticity) allow exploration of new transition patterns while feedback mechanisms guide the exploitation of successful patterns.  
      - These two mechanisms balance innovation and stability, fostering emergent and robust learning.

   - **Role of Connections**:  
     This interplay ensures that the framework’s components work cohesively, modeling meditation as a dynamic and interconnected process. The emergent behaviors resulting from these connections accurately reflect the complexities of real-world cognitive dynamics during meditation.

---
## Simulation Mechanisms: Dynamic Interactions Driving Emergence

### Key Features of Simulation Mechanisms

1. **Activation Dynamics**  
   - **Description**:  
     Activation dynamics govern the competition among thoughtseeds based on their interaction weights, state targets, and noise. This drives the shifts in attention and focus during meditation.  
   - **Implementation**:  
     Found in `thoughtseed_network.py`, the `ThoughtseedNetwork.update` method calculates the activation effects of competing thoughtseeds:
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
   - **Role**:  
     This mechanism enables emergent thought dynamics by simulating competitive interactions among thoughtseeds, influenced by both local interactions and global states.

2. **Meta-Awareness Modulation**  
   - **Description**:  
     Meta-awareness acts as a regulatory mechanism, adjusting the influence of distractions and sustaining meditative focus. It dynamically changes based on habituation, state, and noise.  
   - **Implementation**:  
     Found in `metacognition.py`, the `MetaCognitiveMonitor.update` method modulates awareness based on external and internal feedback:
     ```python
     def update(self, network: ThoughtseedNetwork, state: str, current_dwell: int, timestep: int) -> float:
         base_awareness = self._calculate_base_awareness(state)
         awareness_with_habituation = base_awareness * (1.0 - self.habituation * 0.3)
         final_awareness = max(0.4, min(1.0, awareness_with_habituation + np.random.normal(0, meta_variance)))
         return final_awareness
     ```
   - **Role**:  
     Regulates the balance between distractions and focused attention, contributing to the emergence of higher-order meditative states.

3. **State Transitions**  
   - **Description**:  
     State transitions are driven by emergent patterns of thoughtseed activations and meta-awareness, rather than predefined rules. This ensures flexibility and adaptability in the simulation.  
   - **Implementation**:  
     Found in `run_simulation.py`, the `MeditationSimulation.run` method orchestrates state transitions by invoking the `MeditationStateManager.update` method:
     ```python
     self.state_manager.update(self.network, self.metacognition, t)
     ```
     The state manager leverages `thoughtseed_network.py` features (e.g., distraction levels) and `metacognition.py` meta-awareness detection to guide transitions.  
   - **Role**:  
     Enables emergent transitions between states like `mind_wandering` and `meta_awareness`, reflecting real-world meditation dynamics.

4. **Emergent Properties**  
   - **Description**:  
     The interplay of activation dynamics, meta-awareness modulation, and transitions leads to emergent behaviors such as sustained focus, spontaneous distractions, and recovery to meditative states.  
   - **Implementation**:  
     The `run` method in `run_simulation.py` integrates the above mechanisms to simulate meditation sessions:
     ```python
     def run(self, ...):
         for t in range(simulation_steps):
             self.network.update(current_state, meta_awareness, ...)
             self.metacognition.update(self.network, current_state, dwell_time, t)
             self.state_manager.update(self.network, self.metacognition, t)
     ```
   - **Role**:  
     Produces realistic meditative experiences by allowing higher-order states to emerge from the combined effects of lower-level mechanisms.

5. **Stochasticity in Dynamics**  
   - **Description**:  
     Controlled randomness is introduced to simulate real-world variability in meditation, such as distractions or fluctuations in focus.  
   - **Implementation**:  
     Found throughout `thoughtseed_network.py` and `metacognition.py`, stochastic elements influence both activations and meta-awareness:
     ```python
     final_target = base_targets[ts] + interaction_effects.get(ts, 0.0) + np.random.normal(0, noise_variance)
     ```
   - **Role**:  
     Ensures robustness and diversity in the simulation, preventing overly deterministic behaviors and fostering emergent dynamics.

6. **Integration with Learning Mechanisms**  
   - **Description**:  
     Simulation mechanisms are tightly integrated with learning mechanisms to ensure that feedback from simulated sessions refines the framework’s parameters, creating a feedback loop.  
   - **Implementation**:  
     Found in `learning_thoughtseeds_revised.py`, where weights, interactions, and transitions are updated based on simulation outcomes:
     ```python
     feedback = evaluate_simulation(session_results)
     adjust_weights(feedback)
     refine_transitions(feedback)
     ```
   - **Role**:  
     Aligns learning and simulation, allowing the framework to iteratively improve based on emergent behaviors.

---

**These features collectively drive the dynamics of the Thoughtseeds Framework, enabling it to simulate meditation as an emergent, adaptive process that mirrors real-world cognitive states.**

---

## Why Emergent Phenomena, Not Hard-Coded Rules

- **Dynamic Adaptability**:  
  The Thoughtseeds Framework models meditation as a dynamic process where states (e.g., `mind_wandering`, `meta_awareness`) naturally emerge from the interactions between thoughtseeds, environmental inputs, and meta-awareness. This approach avoids rigid, predefined logic, allowing the simulation to adapt to a variety of scenarios and individual differences.  

- **Realistic Cognitive Dynamics**:  
  Meditation is not a linear or deterministic process. Distractions (e.g., `pending_tasks`) and refocusing efforts (e.g., `breath_control`) occur unpredictably, influenced by internal states and external stimuli. The framework captures this complexity by allowing higher-order states to emerge through activation dynamics and feedback loops, rather than enforcing strict state transitions.

- **Expertise Variation**:  
  Novices and experts exhibit distinct cognitive patterns during meditation. Novices are more prone to distractions, while experts show stronger inhibition of irrelevant thoughts. By relying on emergent phenomena, the framework naturally differentiates these behaviors through learned parameters, removing the need for separate code paths for novices and experts.

- **Examples in the Code**:  
  Emergent phenomena are evident in the following implementations:
  - **Activation Dynamics**: Thoughtseed interactions, influenced by weights and noise, drive competitive dynamics, producing emergent attention shifts.  
    ```python
    final_target = base_targets[ts] + interaction_effects.get(ts, 0.0) + np.random.normal(0, noise_variance)
    ```
  - **State Transitions**: Transition probabilities evolve based on learned patterns, reflecting real-world shifts in meditative states.  
    ```python
    for state in self.transition_counts:
        total_transitions = sum(self.transition_counts[state].values())
        self.transition_probs[state] = {
            next_state: count / total_transitions
            for next_state, count in self.transition_counts[state].items()
        }
    ```

- **Why This Matters**:  
  Relying on emergent phenomena ensures that the simulation mirrors the fluid, non-deterministic nature of meditation, making it robust, scalable, and applicable across diverse use cases.

---

## Locating the Logic

The Thoughtseeds Framework is modular, with specific components handling different aspects of self-organization, learning, and emergence. Here’s where the critical logic resides:

1. **Self-Organization**:  
   - File: `thoughtseed_network.py`  
   - Key Method: `ThoughtseedNetwork.update`  
   - Role: Manages local interactions and activations among thoughtseeds, fostering emergent patterns without centralized control.
     ```python
     def update(self, meditation_state: str, meta_awareness: float, ...):
         ...
         interaction_effects[ts] = sum(strength * current_activations[ts] * 0.1 for other_ts, strength in self.interactions[ts].get("connections", {}).items())
     ```

2. **Learning Mechanisms**:  
   - File: `learning_thoughtseeds_revised.py`  
   - Key Class: `RuleBasedHybridLearner`  
   - Role: Implements adaptive mechanisms to adjust weights, extract causal interactions, and refine transition probabilities based on feedback.  
     ```python
     feedback = evaluate_session(session)
     adjust_weights(feedback)
     refine_transitions(feedback)
     ```

3. **Emergence**:  
   - File: `run_simulation.py`  
   - Key Method: `MeditationSimulation.run`  
   - Role: Orchestrates the interplay of thoughtseed activations, meta-awareness modulation, and state transitions, producing higher-order emergent behaviors.
     ```python
     self.network.update(current_state, meta_awareness, ...)
     self.metacognition.update(self.network, current_state, dwell_time, t)
     self.state_manager.update(self.network, self.metacognition, t)
     ```

4. **Meta-Awareness Regulation**:  
   - File: `metacognition.py`  
   - Key Method: `MetaCognitiveMonitor.update`  
   - Role: Modulates meta-awareness based on habituation, state, and noise, influencing state transitions and focus dynamics.
     ```python
     final_awareness = max(0.4, min(1.0, awareness_with_habituation + np.random.normal(0, meta_variance)))
     ```

---

## Intuition and Design

The Thoughtseeds Framework is designed to mimic the natural dynamics of meditation, blending principles of self-organization, emergence, and adaptive learning. Here’s how intuition and design principles are embedded in the framework:

1. **Agent-Based Modeling**:  
   - Thoughtseeds act as dynamic attentional agents, competing for attention and driving state transitions. This mirrors the real-world cognitive process where various thoughts vie for dominance in the mind. They are not fully independent, but can be said to have **"dependent-origination"** - a core meditation philosophy.  
   - Example: Local interactions among thoughtseeds in `thoughtseed_network.py` create emergent attention dynamics.

2. **Hierarchical Structure**:  
   - The framework integrates three levels—**knowledge domains**, **thoughtseed networks**, and **meta-cognition**—to model meditation as a multiscale process.  
   - Example: Knowledge domains define attractors for thoughtseeds, while meta-cognition modulates their competition, creating a bidirectional feedback loop.

3. **Emergence Over Determinism**:  
   - The framework avoids hardcoded rules, instead allowing states and transitions to emerge from interactions and learned parameters. This approach captures the fluidity and unpredictability of meditation.  
   - Example: State transitions in `run_simulation.py` are driven by dynamic probabilities, not predefined sequences.

4. **Adaptation and Feedback**:  
   - Simulated meditation sessions provide feedback that refines the framework’s parameters, ensuring continuous improvement and alignment with real-world meditation dynamics.  
   - Example: Feedback-driven adaptation in `learning_thoughtseeds_revised.py` adjusts weights and transitions based on session outcomes.

5. **Stochasticity for Realism**:  
   - Controlled randomness is introduced to simulate variability in meditation experiences, such as distractions or fluctuating focus levels.  
   - Example: Noise in `thoughtseed_network.py` influences activations, fostering robustness and diversity in emergent behaviors.

6. **Scalability and Flexibility**:  
   - The modular design allows the framework to scale across different meditation practices and adapt to varying expertise levels.  
   - Example: Expertise-dependent learning in `learning_thoughtseeds_revised.py` differentiates behaviors for novices and experts.

---

By combining these principles, the Thoughtseeds Framework offers a powerful and realistic simulation of meditative states, capable of capturing the nuanced dynamics of cognitive processes during meditation.
