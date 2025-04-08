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