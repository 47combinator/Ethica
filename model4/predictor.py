"""
Predictor for Model 4: Virtue Ethics Moral AI
==============================================
Generates decisions with virtue-based reasoning.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from .virtues import VirtueSystem
from .balancer import VirtueBalancer
from model2.labels import get_label


@dataclass
class VirtueDecision:
    scenario_id: str
    chosen_action: str
    chosen_action_description: str
    confidence: float
    action_scores: List[Dict]        # {action_id, desc, score, is_chosen}
    virtue_profile: Dict[str, float] # Virtue scores for chosen action
    virtue_weights: Dict[str, float] # Context-adjusted weights used
    dominant_virtue: str
    dominant_virtue_name: str
    conflicts: List[Dict]
    context_signals: List[str]
    human_agreement: float


class VirtuePredictor:
    """Uses the virtue balancer to make moral decisions."""
    
    def __init__(self, balancer: VirtueBalancer):
        self.balancer = balancer
        self.virtue_system = balancer.virtue_system
        self.decision_log: List[VirtueDecision] = []
    
    def predict_scenario(self, scenario: Dict) -> VirtueDecision:
        result = self.balancer.evaluate_scenario(scenario)
        
        actions = scenario.get("actions", [])
        best_idx = result["chosen_action_idx"]
        best_action = result["chosen_action"]
        sid = scenario["id"]
        
        # Build action score list
        action_score_list = []
        for idx, (action, score) in enumerate(zip(actions, result["overall_scores"])):
            action_score_list.append({
                "action_id": action["id"],
                "description": action["description"],
                "score": score,
                "is_chosen": idx == best_idx,
                "virtue_profile": result["action_virtue_profiles"][idx],
            })
        
        # Human agreement
        label = get_label(sid)
        agreement = 0.0
        if label:
            if best_idx == label["preferred_action_idx"]:
                agreement = 1.0
            else:
                dist = label.get("distribution", [])
                agreement = dist[best_idx] if best_idx < len(dist) else 0.0
        
        # Dominant virtue name
        dom_v = self.virtue_system.get_virtue(result["dominant_virtue"])
        dom_name = dom_v.name if dom_v else result["dominant_virtue"]
        
        decision = VirtueDecision(
            scenario_id=sid,
            chosen_action=best_action["id"],
            chosen_action_description=best_action["description"],
            confidence=result["confidence"],
            action_scores=action_score_list,
            virtue_profile=result["action_virtue_profiles"][best_idx],
            virtue_weights=result["virtue_weights"],
            dominant_virtue=result["dominant_virtue"],
            dominant_virtue_name=dom_name,
            conflicts=result["conflicts"],
            context_signals=result["context"]["context_signals"],
            human_agreement=agreement,
        )
        self.decision_log.append(decision)
        return decision
    
    def batch_predict(self, scenarios: List[Dict]) -> List[VirtueDecision]:
        return [self.predict_scenario(s) for s in scenarios]
    
    def get_statistics(self) -> Dict:
        if not self.decision_log:
            return {"total": 0}
        
        # Dominant virtue distribution
        virtue_counts = {}
        for d in self.decision_log:
            v = d.dominant_virtue_name
            virtue_counts[v] = virtue_counts.get(v, 0) + 1
        
        return {
            "total_predictions": len(self.decision_log),
            "avg_confidence": round(float(np.mean([d.confidence for d in self.decision_log])), 3),
            "human_agreement_rate": round(float(np.mean([d.human_agreement for d in self.decision_log])), 3),
            "avg_conflicts": round(float(np.mean([len(d.conflicts) for d in self.decision_log])), 1),
            "dominant_virtue_distribution": virtue_counts,
        }
    
    def clear_log(self):
        self.decision_log = []
