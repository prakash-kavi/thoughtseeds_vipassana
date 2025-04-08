THOUGHTSEEDS = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
STATES = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']
STATE_DWELL_TIMES = {
    'novice': {
        'breath_control': (10, 15),
        'mind_wandering': (20, 30),
        'meta_awareness': (2, 5),
        'redirect_breath': (2, 5)
    },
    'expert': {
        'breath_control': (15, 25),
        'mind_wandering': (8, 12),
        'meta_awareness': (1, 3),
        'redirect_breath': (1, 3)
    }
}

# Simulation constants
TRANSITION_SETTINGS = {
    "novice": {
        "breath_control": {
            "spontaneous_chance": 0.03,  # Increased for more variability
            "distraction_threshold": 0.45  # Slightly lower threshold
        },
        "mind_wandering": {
            "detection_base": 0.10,  # Increased for novices
            "detection_growth": 0.03,  # Faster growth in detection probability
            "max_dwell_enforce": 0.95  # High probability to enforce max dwell
        },
        "meta_awareness": {
            "variability": 0.5,  # Higher variability
            "min_transition_prob": 0.3  # Higher minimum probability
        },
        "redirect_breath": {
            "success_threshold": 0.4,  # Easier to transition back
            "success_prob_scale": 0.8  # But with lower success likelihood
        }
    },
    "expert": {
        "breath_control": {
            "spontaneous_chance": 0.015,  # Still occurs but less frequent
            "distraction_threshold": 0.55  # Higher threshold for experts
        },
        "mind_wandering": {
            "detection_base": 0.15,  # Higher base detection for experts
            "detection_growth": 0.04,  # Faster growth in detection probability
            "max_dwell_enforce": 0.98  # Strong enforcement of max dwell
        },
        "meta_awareness": {
            "variability": 0.4,  # Lower but still significant variability
            "min_transition_prob": 0.4  # Higher minimum probability
        },
        "redirect_breath": {
            "success_threshold": 0.45,  # Slightly harder threshold
            "success_prob_scale": 1.0  # But higher success likelihood
        }
    }
}

MEDITATION_STATE_THOUGHTSEED_ATTRACTORS = {
    "breath_control": {"primary": ["breath_focus"], "secondary": ["equanimity"], "condition": "redirect_breath"},
    "mind_wandering": {"primary": ["pain_discomfort"], "secondary": ["pending_tasks"], "condition": "breath_control"},
    "meta_awareness": {"primary": ["self_reflection"], "secondary": [], "condition": "mind_wandering"},
    "redirect_breath": {"primary": ["equanimity"], "secondary": ["breath_focus"], "condition": "meta_awareness"}
}

THOUGHTSEED_AGENTS = {
    "breath_focus": {"id": 0, "category": "focus", "intentional_weights": {"novice": 0.8, "expert": 0.95}, "decay_rate": 0.005, "recovery_rate": 0.06},
    "equanimity": {"id": 4, "category": "emotional_regulation", "intentional_weights": {"novice": 0.3, "expert": 0.8}, "decay_rate": 0.008, "recovery_rate": 0.045},
    "pain_discomfort": {"id": 1, "category": "body_sensation", "intentional_weights": {"novice": 0.6, "expert": 0.3}, "decay_rate": 0.02, "recovery_rate": 0.025},
    "pending_tasks": {"id": 2, "category": "distraction", "intentional_weights": {"novice": 0.7, "expert": 0.2}, "decay_rate": 0.015, "recovery_rate": 0.03},
    "self_reflection": {"id": 3, "category": "meta-awareness", "intentional_weights": {"novice": 0.5, "expert": 0.5}, "decay_rate": 0.003, "recovery_rate": 0.015}
}

THOUGHTSEED_INTERACTIONS = {
    "breath_focus": {"connections": {"pain_discomfort": -0.6, "self_reflection": 0.5, "equanimity": 0.5, "pending_tasks": -0.5}},
    "equanimity": {"connections": {"breath_focus": 0.6, "pain_discomfort": -0.6, "self_reflection": 0.5, "pending_tasks": -0.4}},
    "pain_discomfort": {"connections": {"breath_focus": -0.4, "equanimity": -0.3, "self_reflection": 0.5, "pending_tasks": 0.3}},
    "pending_tasks": {"connections": {"breath_focus": -0.3, "self_reflection": 0.3, "equanimity": -0.2, "pain_discomfort": 0.2}},
    "self_reflection": {"connections": {"breath_focus": 0.3, "equanimity": 0.6, "pain_discomfort": -0.2, "pending_tasks": 0.2}}
}

# Noise parameters for biological variability
NOISE_LEVEL = {'novice': 0.08, 'expert': 0.04}
