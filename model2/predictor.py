"""
Predictor for Model 2: Learning-Based Moral AI
===============================================
Uses trained model to make moral decisions on new scenarios.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .features import FeatureExtractor
from .network import MoralNetwork
from .labels import get_label


@dataclass
class Model2Decision:
    """Decision output from Model 2."""
    scenario_id: str
    chosen_action: str
    chosen_action_description: str
    confidence: float
    action_scores: List[Dict]
    learned_pattern: str
    feature_influences: Dict
    human_agreement: float  # how much this agrees with human label


class MoralPredictor:
    """
    Uses a trained moral learning model to evaluate ethical scenarios.
    """
    
    def __init__(self, model: MoralNetwork, feature_extractor: FeatureExtractor = None):
        self.model = model
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.decision_log: List[Model2Decision] = []
    
    def predict_scenario(self, scenario: Dict) -> Model2Decision:
        """
        Predict the morally preferred action for a scenario.
        """
        actions = scenario.get("actions", [])
        scenario_id = scenario["id"]
        
        # Extract features for each action
        action_features = []
        for idx, action in enumerate(actions):
            feat = self.feature_extractor.extract_action_features(scenario, action, idx)
            action_features.append(feat)
        
        X = np.array(action_features)
        
        # Predict preference scores
        scores = self.model.predict_proba(X)
        
        # Choose highest scoring action
        best_idx = int(np.argmax(scores))
        best_action = actions[best_idx]
        
        # Calculate confidence from score distribution
        if len(scores) > 1:
            sorted_scores = np.sort(scores)[::-1]
            confidence = float(min(1.0, max(0.1, (sorted_scores[0] - sorted_scores[1]) * 2 + 0.3)))
        else:
            confidence = float(scores[0])
        
        # Get feature importance for this prediction
        feat_importance = self.model.get_feature_importance(
            X, self.feature_extractor.feature_names
        )
        top_features = dict(list(feat_importance.items())[:10])
        
        # Determine learned pattern
        pattern = self._identify_pattern(top_features, scenario)
        
        # Check agreement with human labels
        human_label = get_label(scenario_id)
        agreement = 0.0
        if human_label:
            human_pref = human_label["preferred_action_idx"]
            if best_idx == human_pref:
                agreement = 1.0
            else:
                dist = human_label.get("distribution", [])
                if best_idx < len(dist):
                    agreement = dist[best_idx]
        
        # Build action scores list
        action_score_list = []
        for idx, (action, score) in enumerate(zip(actions, scores)):
            action_score_list.append({
                "action_id": action["id"],
                "description": action["description"],
                "score": float(score),
                "is_chosen": idx == best_idx,
            })
        
        decision = Model2Decision(
            scenario_id=scenario_id,
            chosen_action=best_action["id"],
            chosen_action_description=best_action["description"],
            confidence=confidence,
            action_scores=action_score_list,
            learned_pattern=pattern,
            feature_influences=top_features,
            human_agreement=agreement,
        )
        
        self.decision_log.append(decision)
        return decision
    
    def batch_predict(self, scenarios: List[Dict]) -> List[Model2Decision]:
        """Predict for multiple scenarios."""
        return [self.predict_scenario(s) for s in scenarios]
    
    def _identify_pattern(self, features: Dict, scenario: Dict) -> str:
        """Identify the learned moral pattern driving the decision."""
        top_keys = list(features.keys())[:5]
        
        patterns = {
            "harm_to_others": "harm minimization",
            "fairness_impact": "fairness optimization",
            "accountability_score": "accountability seeking",
            "benefit_score": "benefit maximization",
            "welfare_impact": "welfare optimization",
            "lives_at_risk_score": "life preservation",
            "safety_risk": "safety prioritization",
            "discrimination_level": "anti-discrimination",
            "deception_level": "honesty preference",
            "transparency_score": "transparency seeking",
            "privacy_impact": "privacy protection",
            "autonomy_impact": "autonomy respect",
        }
        
        for key in top_keys:
            if key in patterns:
                return patterns[key]
        
        return "complex pattern matching"
    
    def get_statistics(self) -> Dict:
        """Get statistics from prediction log."""
        if not self.decision_log:
            return {"total": 0}
        
        total = len(self.decision_log)
        avg_conf = np.mean([d.confidence for d in self.decision_log])
        avg_agreement = np.mean([d.human_agreement for d in self.decision_log])
        
        patterns = {}
        for d in self.decision_log:
            patterns[d.learned_pattern] = patterns.get(d.learned_pattern, 0) + 1
        
        return {
            "total_predictions": total,
            "average_confidence": round(float(avg_conf), 3),
            "human_agreement_rate": round(float(avg_agreement), 3),
            "pattern_distribution": patterns,
        }
    
    def clear_log(self):
        self.decision_log = []
