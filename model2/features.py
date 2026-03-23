"""
Feature Extractor for Model 2: Learning-Based Moral AI
======================================================
Converts structured ethical scenarios into numerical feature vectors
that a neural network can learn from.
"""

import numpy as np
from typing import Dict, List, Tuple

# All consequence keys used across scenarios
CONSEQUENCE_KEYS = [
    "harm_to_others", "harm_to_self", "lives_at_risk_score",
    "fairness_impact", "safety_risk", "accountability_score",
    "benefit_score", "welfare_impact", "collateral_damage",
    "discrimination_level", "deception_level", "transparency_score",
    "privacy_impact", "autonomy_impact", "consent_violation",
    "human_oversight_maintained", "escalation_possible",
    "proportionality_score", "restrictiveness", "data_exposure",
    "manipulation_level", "legal_violation_score",
]

CATEGORY_LIST = [
    "autonomous_vehicles", "healthcare_ai", "hiring_bias",
    "military_ai", "privacy_surveillance", "financial_ai",
    "disaster_response", "human_ai_interaction",
    "corporate_pressure", "moral_ambiguity",
]

DIMENSION_LIST = [
    "harm", "life_preservation", "fairness", "autonomy", "honesty",
    "privacy", "responsibility", "human_oversight", "proportionality",
    "beneficence", "discrimination", "deception", "legal_compliance",
    "transparency", "consent", "manipulation", "safety", "welfare",
]


class FeatureExtractor:
    """
    Extracts numerical features from ethical scenarios for ML training.
    
    Feature vector structure per action:
    [action_consequences(22) + scenario_context(10 category + 18 dims + 5 meta)] = 55 features
    """
    
    def __init__(self):
        self.n_consequence_features = len(CONSEQUENCE_KEYS)
        self.n_category_features = len(CATEGORY_LIST)
        self.n_dimension_features = len(DIMENSION_LIST)
        self.n_meta_features = 5
        self.feature_size = (
            self.n_consequence_features +
            self.n_category_features +
            self.n_dimension_features +
            self.n_meta_features
        )
        self.feature_names = self._build_feature_names()
    
    def _build_feature_names(self) -> List[str]:
        names = list(CONSEQUENCE_KEYS)
        names += [f"cat_{c}" for c in CATEGORY_LIST]
        names += [f"dim_{d}" for d in DIMENSION_LIST]
        names += ["num_actions", "action_index", "num_dimensions",
                   "avg_harm_across_actions", "max_fairness_diff"]
        return names
    
    def extract_action_features(self, scenario: Dict, action: Dict,
                                 action_idx: int) -> np.ndarray:
        """Extract feature vector for a single action within a scenario."""
        features = []
        
        # 1. Action consequence features (22)
        consequences = action.get("consequences", {})
        for key in CONSEQUENCE_KEYS:
            features.append(consequences.get(key, 0.5))
        
        # 2. Category one-hot (10)
        category = scenario.get("category", "")
        for cat in CATEGORY_LIST:
            features.append(1.0 if category == cat else 0.0)
        
        # 3. Ethical dimension flags (18)
        dims = scenario.get("ethical_dimensions", [])
        for dim in DIMENSION_LIST:
            features.append(1.0 if dim in dims else 0.0)
        
        # 4. Meta features (5)
        actions = scenario.get("actions", [])
        num_actions = len(actions)
        features.append(num_actions / 3.0)  # normalized
        features.append(action_idx / max(num_actions - 1, 1))  # position
        features.append(len(dims) / len(DIMENSION_LIST))  # dim density
        
        # Average harm across all actions
        all_harms = [a.get("consequences", {}).get("harm_to_others", 0.5)
                     for a in actions]
        features.append(np.mean(all_harms))
        
        # Max fairness differential
        all_fairness = [a.get("consequences", {}).get("fairness_impact", 0.5)
                        for a in actions]
        features.append(max(all_fairness) - min(all_fairness) if len(all_fairness) > 1 else 0)
        
        return np.array(features, dtype=np.float64)
    
    def extract_scenario_features(self, scenario: Dict) -> List[Tuple[np.ndarray, int]]:
        """
        Extract features for all actions in a scenario.
        Returns list of (feature_vector, action_index) tuples.
        """
        actions = scenario.get("actions", [])
        result = []
        for idx, action in enumerate(actions):
            feat = self.extract_action_features(scenario, action, idx)
            result.append((feat, idx))
        return result
    
    def extract_dataset(self, scenarios: List[Dict],
                         labels: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract full training dataset from scenarios and human labels.
        
        Args:
            scenarios: List of scenario dicts
            labels: Dict mapping scenario_id -> preferred action index
            
        Returns:
            X: feature matrix (n_samples, n_features)
            y: binary labels (1 = preferred, 0 = not preferred)
            scenario_ids: list of scenario IDs for each sample
        """
        X_list = []
        y_list = []
        sid_list = []
        
        for scenario in scenarios:
            sid = scenario["id"]
            if sid not in labels:
                continue
            
            preferred_idx = labels[sid]["preferred_action_idx"]
            actions = scenario.get("actions", [])
            
            for idx, action in enumerate(actions):
                feat = self.extract_action_features(scenario, action, idx)
                X_list.append(feat)
                y_list.append(1.0 if idx == preferred_idx else 0.0)
                sid_list.append(sid)
        
        return np.array(X_list), np.array(y_list), sid_list
    
    def extract_with_distribution(self, scenarios: List[Dict],
                                   labels: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract dataset with continuous preference scores instead of binary labels.
        Uses the human judgment distribution percentages.
        """
        X_list = []
        y_list = []
        sid_list = []
        
        for scenario in scenarios:
            sid = scenario["id"]
            if sid not in labels:
                continue
            
            distribution = labels[sid].get("distribution", [])
            actions = scenario.get("actions", [])
            
            for idx, action in enumerate(actions):
                feat = self.extract_action_features(scenario, action, idx)
                X_list.append(feat)
                # Use distribution percentage as regression target
                score = distribution[idx] if idx < len(distribution) else 0.0
                y_list.append(score)
                sid_list.append(sid)
        
        return np.array(X_list), np.array(y_list), sid_list
