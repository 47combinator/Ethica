"""
Human Feedback System for Model 3: RLHF Moral AI
=================================================
Simulates human feedback by ranking model responses.
Uses the human judgment labels as ground truth for ranking.
"""

import numpy as np
from typing import Dict, List, Tuple
from model2.labels import get_label, get_all_labels


class HumanFeedbackSystem:
    """
    Simulates human feedback for RLHF training.
    
    In real research, this would be replaced with actual crowdsourced
    human rankings. For this implementation, we simulate feedback using
    the AMR-220 human judgment labels with controlled noise.
    """
    
    def __init__(self, noise_level: float = 0.1, seed: int = 42):
        """
        Args:
            noise_level: How noisy human feedback is (0 = perfect, 1 = random).
                         Simulates real-world disagreement among annotators.
        """
        self.noise_level = noise_level
        self.labels = get_all_labels()
        self.feedback_log = []
        np.random.seed(seed)
    
    def rank_responses(self, scenario: Dict,
                        responses: List[Dict]) -> List[Dict]:
        """
        Simulate human ranking of model responses.
        
        Returns responses sorted from best to worst with ranking scores.
        Each response gets a 'human_rank' and 'human_score' field.
        """
        sid = scenario["id"]
        label = self.labels.get(sid)
        
        if not label:
            # No label available - random ranking with slight preference for first
            for i, r in enumerate(responses):
                r["human_score"] = float(np.random.uniform(0.3, 0.7))
                r["human_rank"] = i + 1
            return responses
        
        preferred_idx = label["preferred_action_idx"]
        distribution = label.get("distribution", [])
        
        # Score each response based on human preference distribution
        for r in responses:
            action_idx = r["action_idx"]
            
            # Base score from human distribution
            if action_idx < len(distribution):
                base_score = distribution[action_idx]
            else:
                base_score = 0.1
            
            # Add noise to simulate annotator disagreement
            noise = np.random.normal(0, self.noise_level)
            score = np.clip(base_score + noise, 0.0, 1.0)
            
            r["human_score"] = float(score)
            r["is_human_preferred"] = (action_idx == preferred_idx)
        
        # Rank by human score (highest = rank 1)
        responses.sort(key=lambda x: -x["human_score"])
        for i, r in enumerate(responses):
            r["human_rank"] = i + 1
        
        return responses
    
    def generate_pairwise_comparisons(self, scenario: Dict,
                                       responses: List[Dict]) -> List[Dict]:
        """
        Generate pairwise preference comparisons from rankings.
        Returns list of {better, worse, margin} pairs.
        """
        ranked = self.rank_responses(scenario, responses)
        pairs = []
        
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                better = ranked[i]  # higher ranked
                worse = ranked[j]   # lower ranked
                margin = better["human_score"] - worse["human_score"]
                
                pairs.append({
                    "scenario_id": scenario["id"],
                    "better_action": better["action_id"],
                    "worse_action": worse["action_id"],
                    "better_features": better["features"],
                    "worse_features": worse["features"],
                    "better_score": better["human_score"],
                    "worse_score": worse["human_score"],
                    "margin": float(margin),
                })
        
        self.feedback_log.extend(pairs)
        return pairs
    
    def collect_batch_feedback(self, scenarios: List[Dict],
                                base_model) -> Tuple[List[Dict], Dict]:
        """
        Collect feedback for a batch of scenarios.
        
        Returns:
            all_pairs: List of pairwise comparisons
            stats: Feedback collection statistics
        """
        all_pairs = []
        total_responses = 0
        
        for scenario in scenarios:
            responses = base_model.generate_responses(scenario)
            total_responses += len(responses)
            pairs = self.generate_pairwise_comparisons(scenario, responses)
            all_pairs.extend(pairs)
        
        stats = {
            "scenarios_evaluated": len(scenarios),
            "total_responses": total_responses,
            "total_pairs": len(all_pairs),
            "avg_margin": float(np.mean([p["margin"] for p in all_pairs])) if all_pairs else 0,
        }
        
        return all_pairs, stats
    
    def get_feedback_summary(self) -> Dict:
        """Get summary of all collected feedback."""
        if not self.feedback_log:
            return {"total_pairs": 0}
        
        margins = [p["margin"] for p in self.feedback_log]
        return {
            "total_pairs": len(self.feedback_log),
            "avg_margin": round(float(np.mean(margins)), 3),
            "clear_preferences": sum(1 for m in margins if m > 0.2),
            "close_calls": sum(1 for m in margins if m <= 0.2),
        }
