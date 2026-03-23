"""
Predictor for Model 3: RLHF Moral AI
=====================================
Uses the RLHF-aligned policy to make moral decisions.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from model2.features import FeatureExtractor
from model2.labels import get_label


@dataclass
class RLHFDecision:
    """Decision output from Model 3."""
    scenario_id: str
    chosen_action: str
    chosen_action_description: str
    confidence: float
    action_scores: List[Dict]
    reward_scores: List[float]
    avg_reward: float
    learned_pattern: str
    human_agreement: float
    sycophancy_risk: float


class RLHFPredictor:
    """Uses RLHF-trained model to make moral decisions."""
    
    def __init__(self, policy_model, reward_model, feature_extractor=None):
        self.policy = policy_model
        self.reward = reward_model
        self.feat = feature_extractor or FeatureExtractor()
        self.decision_log: List[RLHFDecision] = []
    
    def predict_scenario(self, scenario: Dict) -> RLHFDecision:
        actions = scenario.get("actions", [])
        sid = scenario["id"]
        
        features_list = []
        for idx, action in enumerate(actions):
            feat = self.feat.extract_action_features(scenario, action, idx)
            features_list.append(feat)
        
        X = np.array(features_list)
        
        # Policy scores (preference)
        policy_scores = self.policy.predict(X)
        
        # Reward model scores
        reward_scores = self.reward.predict_reward(X) if self.reward.is_trained else policy_scores.copy()
        
        # Choose action with highest policy score
        best_idx = int(np.argmax(policy_scores))
        best_action = actions[best_idx]
        
        # Confidence
        if len(policy_scores) > 1:
            s = np.sort(policy_scores)[::-1]
            confidence = float(min(1.0, max(0.1, (s[0] - s[1]) * 2 + 0.3)))
        else:
            confidence = float(policy_scores[0])
        
        # Human agreement
        label = get_label(sid)
        agreement = 0.0
        if label:
            if best_idx == label["preferred_action_idx"]:
                agreement = 1.0
            else:
                dist = label.get("distribution", [])
                agreement = dist[best_idx] if best_idx < len(dist) else 0.0
        
        # Sycophancy risk: does the model pick what humans like even when
        # reward model suggests something different?
        reward_best = int(np.argmax(reward_scores))
        syc_risk = 0.0
        if label and reward_best != label["preferred_action_idx"] and best_idx == label["preferred_action_idx"]:
            syc_risk = 0.8
        elif reward_best != best_idx:
            syc_risk = 0.3
        
        # Learned pattern
        pattern = self._identify_pattern(X, best_idx, scenario)
        
        action_score_list = []
        for idx, (action, ps, rs) in enumerate(zip(actions, policy_scores, reward_scores)):
            action_score_list.append({
                "action_id": action["id"],
                "description": action["description"],
                "policy_score": float(ps),
                "reward_score": float(rs),
                "is_chosen": idx == best_idx,
            })
        
        decision = RLHFDecision(
            scenario_id=sid,
            chosen_action=best_action["id"],
            chosen_action_description=best_action["description"],
            confidence=confidence,
            action_scores=action_score_list,
            reward_scores=[float(r) for r in reward_scores],
            avg_reward=float(np.mean(reward_scores)),
            learned_pattern=pattern,
            human_agreement=agreement,
            sycophancy_risk=syc_risk,
        )
        self.decision_log.append(decision)
        return decision
    
    def batch_predict(self, scenarios: List[Dict]) -> List[RLHFDecision]:
        return [self.predict_scenario(s) for s in scenarios]
    
    def _identify_pattern(self, X, best_idx, scenario):
        """Identify the RLHF-learned pattern."""
        consequences = scenario["actions"][best_idx].get("consequences", {})
        patterns = [
            ("harm_to_others", 0.3, "human-approved harm minimization"),
            ("accountability_score", 0.6, "feedback-aligned accountability"),
            ("fairness_impact", 0.6, "reward-optimized fairness"),
            ("benefit_score", 0.6, "human-preferred benefit"),
            ("deception_level", 0.3, "honesty alignment"),
        ]
        for key, thresh, name in patterns:
            val = consequences.get(key, 0.5)
            if key in ("harm_to_others", "deception_level"):
                if val < thresh:
                    return name
            else:
                if val > thresh:
                    return name
        return "reward-maximizing policy"
    
    def get_statistics(self) -> Dict:
        if not self.decision_log:
            return {"total": 0}
        total = len(self.decision_log)
        return {
            "total_predictions": total,
            "avg_confidence": round(float(np.mean([d.confidence for d in self.decision_log])), 3),
            "human_agreement_rate": round(float(np.mean([d.human_agreement for d in self.decision_log])), 3),
            "avg_sycophancy_risk": round(float(np.mean([d.sycophancy_risk for d in self.decision_log])), 3),
            "avg_reward": round(float(np.mean([d.avg_reward for d in self.decision_log])), 3),
        }
    
    def clear_log(self):
        self.decision_log = []
