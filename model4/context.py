"""
Context Analyzer for Model 4: Virtue Ethics Moral AI
=====================================================
Understands scenario context to dynamically adjust virtue weights.
This implements Aristotle's concept of phronesis (practical wisdom).
"""

import numpy as np
from typing import Dict, List, Tuple


# Context profiles: which virtues matter more in which domains
CONTEXT_PROFILES = {
    "autonomous_vehicles": {
        "V_COMP": 1.3, "V_JUST": 1.1, "V_RESP": 1.2, "V_PRUD": 1.4,
        "V_COUR": 0.8, "V_HONE": 0.9, "V_TEMP": 1.0, "V_BENE": 1.1,
        "urgency": "high", "stakes": "life",
    },
    "healthcare_ai": {
        "V_COMP": 1.4, "V_JUST": 1.2, "V_RESP": 1.3, "V_PRUD": 1.1,
        "V_COUR": 0.9, "V_HONE": 1.3, "V_TEMP": 0.8, "V_BENE": 1.3,
        "urgency": "high", "stakes": "life",
    },
    "hiring_bias": {
        "V_COMP": 0.9, "V_JUST": 1.5, "V_RESP": 1.1, "V_PRUD": 1.0,
        "V_COUR": 1.1, "V_HONE": 1.2, "V_TEMP": 1.0, "V_BENE": 1.0,
        "urgency": "low", "stakes": "welfare",
    },
    "military_ai": {
        "V_COMP": 1.2, "V_JUST": 1.3, "V_RESP": 1.4, "V_PRUD": 1.3,
        "V_COUR": 1.2, "V_HONE": 0.9, "V_TEMP": 1.3, "V_BENE": 0.8,
        "urgency": "high", "stakes": "life",
    },
    "privacy_surveillance": {
        "V_COMP": 0.8, "V_JUST": 1.3, "V_RESP": 1.1, "V_PRUD": 1.2,
        "V_COUR": 1.0, "V_HONE": 1.4, "V_TEMP": 1.2, "V_BENE": 0.9,
        "urgency": "medium", "stakes": "rights",
    },
    "financial_ai": {
        "V_COMP": 0.9, "V_JUST": 1.4, "V_RESP": 1.3, "V_PRUD": 1.2,
        "V_COUR": 1.1, "V_HONE": 1.3, "V_TEMP": 1.1, "V_BENE": 1.0,
        "urgency": "medium", "stakes": "welfare",
    },
    "disaster_response": {
        "V_COMP": 1.4, "V_JUST": 1.0, "V_RESP": 1.3, "V_PRUD": 1.3,
        "V_COUR": 1.3, "V_HONE": 0.8, "V_TEMP": 0.7, "V_BENE": 1.2,
        "urgency": "critical", "stakes": "life",
    },
    "human_ai_interaction": {
        "V_COMP": 1.2, "V_JUST": 1.0, "V_RESP": 1.1, "V_PRUD": 1.1,
        "V_COUR": 0.9, "V_HONE": 1.4, "V_TEMP": 1.2, "V_BENE": 1.3,
        "urgency": "low", "stakes": "welfare",
    },
    "corporate_pressure": {
        "V_COMP": 0.8, "V_JUST": 1.2, "V_RESP": 1.3, "V_PRUD": 1.0,
        "V_COUR": 1.5, "V_HONE": 1.4, "V_TEMP": 0.9, "V_BENE": 0.9,
        "urgency": "medium", "stakes": "integrity",
    },
    "moral_ambiguity": {
        "V_COMP": 1.1, "V_JUST": 1.1, "V_RESP": 1.1, "V_PRUD": 1.4,
        "V_COUR": 1.0, "V_HONE": 1.1, "V_TEMP": 1.2, "V_BENE": 1.0,
        "urgency": "variable", "stakes": "mixed",
    },
}

# Urgency multipliers for courage and prudence
URGENCY_MODIFIERS = {
    "critical": {"V_COUR": 1.3, "V_PRUD": 1.2, "V_TEMP": 0.7},
    "high":     {"V_COUR": 1.1, "V_PRUD": 1.1, "V_TEMP": 0.9},
    "medium":   {"V_COUR": 1.0, "V_PRUD": 1.0, "V_TEMP": 1.0},
    "low":      {"V_COUR": 0.8, "V_PRUD": 0.9, "V_TEMP": 1.2},
    "variable": {"V_COUR": 1.0, "V_PRUD": 1.1, "V_TEMP": 1.0},
}


class ContextAnalyzer:
    """
    Analyzes scenario context to determine how virtues should be weighted.
    Implements phronesis (practical wisdom) through context-aware reasoning.
    """
    
    def __init__(self):
        self.profiles = CONTEXT_PROFILES
    
    def analyze_context(self, scenario: Dict) -> Dict:
        """
        Analyze a scenario and determine context-adjusted virtue weights.
        
        Returns:
            {
                "category": str,
                "virtue_weights": {virtue_id: adjusted_weight},
                "urgency": str,
                "stakes": str,
                "ethical_complexity": float,
                "context_signals": list,
            }
        """
        category = scenario.get("category", "moral_ambiguity")
        dimensions = scenario.get("ethical_dimensions", [])
        actions = scenario.get("actions", [])
        
        # Get base profile
        profile = self.profiles.get(category, self.profiles["moral_ambiguity"])
        urgency = profile.get("urgency", "medium")
        stakes = profile.get("stakes", "mixed")
        
        # Start with profile weights
        weights = {}
        for vid in ["V_COMP", "V_JUST", "V_HONE", "V_RESP", "V_COUR", "V_PRUD", "V_TEMP", "V_BENE"]:
            weights[vid] = profile.get(vid, 1.0)
        
        # Apply urgency modifiers
        urgency_mods = URGENCY_MODIFIERS.get(urgency, {})
        for vid, mod in urgency_mods.items():
            if vid in weights:
                weights[vid] *= mod
        
        # Adjust based on ethical dimensions present
        context_signals = []
        
        if "harm" in dimensions or "life_preservation" in dimensions:
            weights["V_COMP"] *= 1.15
            weights["V_BENE"] *= 1.1
            context_signals.append("life/harm scenario -> compassion boosted")
        
        if "fairness" in dimensions or "discrimination" in dimensions:
            weights["V_JUST"] *= 1.2
            context_signals.append("fairness at stake -> justice boosted")
        
        if "deception" in dimensions or "honesty" in dimensions:
            weights["V_HONE"] *= 1.2
            context_signals.append("honesty dimension -> honesty boosted")
        
        if "privacy" in dimensions:
            weights["V_JUST"] *= 1.1
            weights["V_HONE"] *= 1.1
            context_signals.append("privacy concern -> justice and honesty boosted")
        
        if "manipulation" in dimensions:
            weights["V_COUR"] *= 1.3
            weights["V_HONE"] *= 1.2
            context_signals.append("manipulation detected -> courage and honesty boosted")
        
        if "consent" in dimensions or "autonomy" in dimensions:
            weights["V_JUST"] *= 1.1
            weights["V_RESP"] *= 1.1
            context_signals.append("autonomy concern -> justice and responsibility boosted")
        
        # Calculate ethical complexity from action consequence variance
        complexity = self._compute_complexity(actions)
        if complexity > 0.6:
            weights["V_PRUD"] *= 1.2
            context_signals.append("high complexity -> prudence boosted")
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: round(v / total * len(weights), 3) for k, v in weights.items()}
        
        return {
            "category": category,
            "virtue_weights": weights,
            "urgency": urgency,
            "stakes": stakes,
            "ethical_complexity": round(complexity, 3),
            "context_signals": context_signals,
            "n_dimensions": len(dimensions),
        }
    
    def _compute_complexity(self, actions: List[Dict]) -> float:
        """Compute ethical complexity from action consequence variance."""
        if len(actions) < 2:
            return 0.3
        
        all_scores = []
        for action in actions:
            cons = action.get("consequences", {})
            scores = list(cons.values())
            if scores:
                all_scores.append(np.mean(scores))
        
        if len(all_scores) < 2:
            return 0.3
        
        # Higher variance between actions = more complex decision
        variance = float(np.std(all_scores))
        # Also factor in number of actions
        action_factor = min(len(actions) / 3.0, 1.0)
        
        return min(1.0, variance * 3 + action_factor * 0.2)
