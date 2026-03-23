"""
Virtue Balancing Engine for Model 4: Virtue Ethics Moral AI
============================================================
Balances competing virtues using context-adjusted weights
and learned virtue application patterns.
"""

import numpy as np
from typing import Dict, List, Tuple
from .virtues import VirtueSystem, Virtue
from .context import ContextAnalyzer
from model2.labels import get_label


class VirtueBalancer:
    """
    Core decision engine that balances virtues to choose the most
    virtuous action. Combines top-down virtue structure with
    bottom-up learning from human judgment patterns.
    """
    
    def __init__(self, seed: int = 42):
        self.virtue_system = VirtueSystem()
        self.context_analyzer = ContextAnalyzer()
        np.random.seed(seed)
        
        # Learned adjustments from training data
        self.learned_adjustments: Dict[str, Dict[str, float]] = {}
        self.is_trained = False
        self.training_stats = {}
    
    def train_from_judgments(self, scenarios: List[Dict]) -> Dict:
        """
        Learn virtue application patterns from human judgment data.
        Adjusts virtue weights based on which virtues align with
        human-preferred actions across the dataset.
        """
        labels = {}
        from model2.labels import get_all_labels
        all_labels = get_all_labels()
        
        # For each category, learn which virtues predict human preferences
        category_virtue_alignment = {}
        correct_before = 0
        correct_after = 0
        total = 0
        
        for scenario in scenarios:
            sid = scenario["id"]
            label = all_labels.get(sid)
            if not label:
                continue
            
            cat = scenario.get("category", "unknown")
            preferred_idx = label["preferred_action_idx"]
            actions = scenario.get("actions", [])
            
            if cat not in category_virtue_alignment:
                category_virtue_alignment[cat] = {
                    vid: [] for vid in self.virtue_system.virtues
                }
            
            # Score each action's virtues
            for idx, action in enumerate(actions):
                cons = action.get("consequences", {})
                virtue_scores = self.virtue_system.score_action_all_virtues(cons)
                
                is_preferred = (idx == preferred_idx)
                
                for vid, score in virtue_scores.items():
                    if is_preferred:
                        category_virtue_alignment[cat][vid].append(score)
                    else:
                        category_virtue_alignment[cat][vid].append(-score)
            
            # Check pre-training accuracy
            total += 1
            context = self.context_analyzer.analyze_context(scenario)
            scores = self._score_actions(actions, context["virtue_weights"])
            if np.argmax(scores) == preferred_idx:
                correct_before += 1
        
        # Compute learned adjustments per category
        for cat, virtue_data in category_virtue_alignment.items():
            self.learned_adjustments[cat] = {}
            for vid, alignment_scores in virtue_data.items():
                if alignment_scores:
                    # Positive mean = this virtue aligns with human preferences
                    alignment = float(np.mean(alignment_scores))
                    # Convert to a weight modifier (0.7 to 1.3)
                    modifier = 1.0 + np.clip(alignment * 0.5, -0.3, 0.3)
                    self.learned_adjustments[cat][vid] = round(modifier, 3)
        
        # Check post-training accuracy
        self.is_trained = True
        for scenario in scenarios:
            sid = scenario["id"]
            label = all_labels.get(sid)
            if not label:
                continue
            actions = scenario.get("actions", [])
            preferred_idx = label["preferred_action_idx"]
            context = self.context_analyzer.analyze_context(scenario)
            weights = self._apply_learned_adjustments(
                context["virtue_weights"], scenario.get("category", "")
            )
            scores = self._score_actions(actions, weights)
            if np.argmax(scores) == preferred_idx:
                correct_after += 1
        
        self.training_stats = {
            "categories_learned": len(self.learned_adjustments),
            "accuracy_before": round(correct_before / max(total, 1), 3),
            "accuracy_after": round(correct_after / max(total, 1), 3),
            "total_scenarios": total,
            "improvement": round(
                (correct_after - correct_before) / max(total, 1), 3
            ),
        }
        
        return self.training_stats
    
    def evaluate_scenario(self, scenario: Dict) -> Dict:
        """
        Evaluate a scenario using virtue ethics framework.
        
        Returns:
            {
                "chosen_action_idx": int,
                "chosen_action": dict,
                "action_virtue_profiles": list of virtue score dicts,
                "virtue_weights": context-adjusted weights,
                "overall_scores": list of final scores,
                "conflicts": list of virtue conflicts,
                "context": context analysis,
                "dominant_virtue": str,
                "confidence": float,
            }
        """
        actions = scenario.get("actions", [])
        
        # 1. Analyze context
        context = self.context_analyzer.analyze_context(scenario)
        
        # 2. Get virtue weights (with learned adjustments)
        weights = context["virtue_weights"].copy()
        if self.is_trained:
            weights = self._apply_learned_adjustments(
                weights, scenario.get("category", "")
            )
        
        # 3. Score each action against all virtues
        action_profiles = []
        all_conflicts = []
        
        for action in actions:
            cons = action.get("consequences", {})
            virtue_scores = self.virtue_system.score_action_all_virtues(cons)
            action_profiles.append(virtue_scores)
            
            conflicts = self.virtue_system.detect_virtue_conflicts(virtue_scores)
            all_conflicts.extend(conflicts)
        
        # 4. Compute weighted overall scores
        overall_scores = self._score_actions(actions, weights)
        
        # 5. Choose best action
        best_idx = int(np.argmax(overall_scores))
        
        # 6. Identify dominant virtue
        best_profile = action_profiles[best_idx]
        weighted_profile = {
            vid: best_profile[vid] * weights.get(vid, 1.0)
            for vid in best_profile
        }
        dominant_vid = max(weighted_profile, key=weighted_profile.get)
        
        # 7. Compute confidence
        sorted_scores = sorted(overall_scores, reverse=True)
        if len(sorted_scores) > 1:
            gap = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, max(0.1, gap * 3 + 0.3))
        else:
            confidence = sorted_scores[0] if sorted_scores else 0.5
        
        # Deduplicate conflicts
        unique = {}
        for c in all_conflicts:
            key = (c["virtue_a"], c["virtue_b"])
            if key not in unique or c["tension"] > unique[key]["tension"]:
                unique[key] = c
        
        return {
            "chosen_action_idx": best_idx,
            "chosen_action": actions[best_idx],
            "action_virtue_profiles": action_profiles,
            "virtue_weights": weights,
            "overall_scores": [round(s, 4) for s in overall_scores],
            "conflicts": list(unique.values())[:8],
            "context": context,
            "dominant_virtue": dominant_vid,
            "confidence": round(confidence, 3),
        }
    
    def _score_actions(self, actions: List[Dict],
                        weights: Dict[str, float]) -> List[float]:
        """Compute weighted virtue scores for all actions."""
        scores = []
        for action in actions:
            cons = action.get("consequences", {})
            virtue_scores = self.virtue_system.score_action_all_virtues(cons)
            
            total = 0
            for vid, vs in virtue_scores.items():
                w = weights.get(vid, 1.0)
                total += vs * w
            
            scores.append(total)
        return scores
    
    def _apply_learned_adjustments(self, weights: Dict[str, float],
                                     category: str) -> Dict[str, float]:
        """Apply learned adjustments for a category."""
        adjustments = self.learned_adjustments.get(category, {})
        adjusted = {}
        for vid, w in weights.items():
            mod = adjustments.get(vid, 1.0)
            adjusted[vid] = round(w * mod, 3)
        return adjusted
