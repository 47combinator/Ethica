"""
Evaluation System for Model 1: Rule-Based Moral AI
===================================================

Provides metrics and analysis tools for evaluating
the moral reasoning quality of the rule-based system.
"""

from typing import List, Dict, Optional
from collections import Counter
import json
import os
from datetime import datetime


class EvaluationSystem:
    """
    Evaluates the moral decision engine across multiple metrics:
    - Moral Consistency
    - Harm Minimization
    - Fairness
    - Rule Conflict Rate
    - Transparency Quality
    - Confidence Distribution
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_decisions(self, decisions: List, scenarios: List[Dict]) -> Dict:
        """
        Run full evaluation on a set of decisions.
        
        Args:
            decisions: List of MoralDecision objects
            scenarios: Corresponding list of scenario dictionaries
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if not decisions:
            return {"error": "No decisions to evaluate"}
        
        scenario_map = {s["id"]: s for s in scenarios}
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(decisions),
            "moral_consistency": self._evaluate_consistency(decisions, scenario_map),
            "harm_minimization": self._evaluate_harm_minimization(decisions, scenario_map),
            "fairness": self._evaluate_fairness(decisions, scenario_map),
            "conflict_analysis": self._evaluate_conflicts(decisions),
            "transparency": self._evaluate_transparency(decisions),
            "confidence_distribution": self._evaluate_confidence(decisions),
            "category_analysis": self._evaluate_by_category(decisions, scenario_map),
            "rule_usage": self._evaluate_rule_usage(decisions),
        }
        
        # Calculate overall score
        metrics["overall_score"] = self._calculate_overall_score(metrics)
        
        self.results.append(metrics)
        return metrics
    
    def _evaluate_consistency(self, decisions: List, scenario_map: Dict) -> Dict:
        """
        Evaluate moral consistency: does the model make similar 
        decisions for similar scenarios?
        """
        # Group decisions by category
        category_decisions = {}
        for d in decisions:
            scenario = scenario_map.get(d.scenario_id, {})
            cat = scenario.get("category", "unknown")
            if cat not in category_decisions:
                category_decisions[cat] = []
            category_decisions[cat].append(d)
        
        # Check if dominant rules are consistent within categories
        consistency_scores = {}
        for cat, cat_decisions in category_decisions.items():
            dominant_rules = [d.dominant_rule for d in cat_decisions]
            rule_counter = Counter(dominant_rules)
            most_common_count = rule_counter.most_common(1)[0][1] if rule_counter else 0
            # Consistency = fraction of decisions using the most common rule
            consistency = most_common_count / len(cat_decisions) if cat_decisions else 0
            consistency_scores[cat] = round(consistency, 3)
        
        avg_consistency = (
            sum(consistency_scores.values()) / len(consistency_scores)
            if consistency_scores else 0
        )
        
        return {
            "average_consistency": round(avg_consistency, 3),
            "category_consistency": consistency_scores,
            "interpretation": self._interpret_consistency(avg_consistency),
        }
    
    def _evaluate_harm_minimization(self, decisions: List, scenario_map: Dict) -> Dict:
        """Evaluate how well the model minimizes harm."""
        harm_scores = []
        
        for d in decisions:
            # Check if chosen action has lower harm than alternatives
            chosen = None
            alternatives = []
            for action_score in d.action_scores:
                harm_rules = ["R03", "R04", "R05"]
                harm_score = sum(
                    action_score.rule_scores.get(r, 0) for r in harm_rules
                ) / len(harm_rules)
                
                if action_score.action_id == d.chosen_action:
                    chosen = harm_score
                else:
                    alternatives.append(harm_score)
            
            if chosen is not None and alternatives:
                # Score = 1 if chosen has best harm score, 0 if worst
                max_alt = max(alternatives)
                min_alt = min(alternatives)
                if max_alt != min_alt:
                    normalized = (chosen - min_alt) / (max_alt - min_alt)
                else:
                    normalized = 0.5
                harm_scores.append(max(0, min(1, normalized)))
        
        avg_harm_min = sum(harm_scores) / len(harm_scores) if harm_scores else 0
        
        return {
            "average_harm_minimization": round(avg_harm_min, 3),
            "scenarios_minimized_harm": sum(1 for s in harm_scores if s > 0.5),
            "total_evaluated": len(harm_scores),
            "interpretation": self._interpret_harm(avg_harm_min),
        }
    
    def _evaluate_fairness(self, decisions: List, scenario_map: Dict) -> Dict:
        """Evaluate fairness across decisions."""
        fairness_scores = []
        
        for d in decisions:
            chosen = None
            for action_score in d.action_scores:
                if action_score.action_id == d.chosen_action:
                    fairness_rules = ["R06", "R07", "R08"]
                    chosen = sum(
                        action_score.rule_scores.get(r, 0) for r in fairness_rules
                    ) / len(fairness_rules)
                    break
            
            if chosen is not None:
                # Normalize to 0-1 (original is -1 to 1)
                normalized = (chosen + 1) / 2
                fairness_scores.append(normalized)
        
        avg_fairness = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0
        
        return {
            "average_fairness": round(avg_fairness, 3),
            "high_fairness_decisions": sum(1 for s in fairness_scores if s > 0.6),
            "low_fairness_decisions": sum(1 for s in fairness_scores if s < 0.4),
            "interpretation": self._interpret_fairness(avg_fairness),
        }
    
    def _evaluate_conflicts(self, decisions: List) -> Dict:
        """Analyze rule conflicts across all decisions."""
        total_conflicts = sum(len(d.conflicts_detected) for d in decisions)
        decisions_with_conflicts = sum(1 for d in decisions if d.conflicts_detected)
        
        # Track which rule pairs conflict most
        conflict_pairs = Counter()
        for d in decisions:
            for conflict in d.conflicts_detected:
                pair = tuple(sorted([conflict["rule_a"], conflict["rule_b"]]))
                conflict_pairs[pair] += 1
        
        top_conflicts = []
        for pair, count in conflict_pairs.most_common(10):
            top_conflicts.append({
                "rules": list(pair),
                "count": count,
            })
        
        return {
            "total_conflicts": total_conflicts,
            "decisions_with_conflicts": decisions_with_conflicts,
            "conflict_rate": round(
                decisions_with_conflicts / len(decisions) if decisions else 0, 3
            ),
            "avg_conflicts_per_decision": round(
                total_conflicts / len(decisions) if decisions else 0, 2
            ),
            "top_conflicting_pairs": top_conflicts,
        }
    
    def _evaluate_transparency(self, decisions: List) -> Dict:
        """Evaluate explanation quality/transparency."""
        reasoning_lengths = []
        has_dominant_rule = 0
        has_conflicts_explained = 0
        has_factors = 0
        
        for d in decisions:
            reasoning_lengths.append(len(d.reasoning_chain))
            if d.dominant_rule:
                has_dominant_rule += 1
            if d.conflicts_detected:
                has_conflicts_explained += 1
            if d.ethical_factors:
                has_factors += 1
        
        avg_reasoning = (
            sum(reasoning_lengths) / len(reasoning_lengths) 
            if reasoning_lengths else 0
        )
        
        # Transparency score: based on completeness of explanations
        components = [
            has_dominant_rule / len(decisions) if decisions else 0,
            min(1.0, avg_reasoning / 7),  # Expect ~7 reasoning steps
            has_factors / len(decisions) if decisions else 0,
        ]
        transparency_score = sum(components) / len(components)
        
        return {
            "average_reasoning_steps": round(avg_reasoning, 1),
            "transparency_score": round(transparency_score, 3),
            "decisions_with_clear_dominant_rule": has_dominant_rule,
            "decisions_with_factor_analysis": has_factors,
        }
    
    def _evaluate_confidence(self, decisions: List) -> Dict:
        """Analyze confidence distribution."""
        confidences = [d.confidence for d in decisions]
        
        if not confidences:
            return {"error": "No confidence data"}
        
        avg = sum(confidences) / len(confidences)
        high = sum(1 for c in confidences if c > 0.7)
        medium = sum(1 for c in confidences if 0.4 <= c <= 0.7)
        low = sum(1 for c in confidences if c < 0.4)
        
        return {
            "average_confidence": round(avg, 3),
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence": low,
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
        }
    
    def _evaluate_by_category(self, decisions: List, scenario_map: Dict) -> Dict:
        """Evaluate performance by scenario category."""
        category_stats = {}
        
        for d in decisions:
            scenario = scenario_map.get(d.scenario_id, {})
            cat = scenario.get("category", "unknown")
            
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "avg_conflicts": 0,
                    "confidences": [],
                    "conflict_counts": [],
                }
            
            category_stats[cat]["count"] += 1
            category_stats[cat]["confidences"].append(d.confidence)
            category_stats[cat]["conflict_counts"].append(len(d.conflicts_detected))
        
        # Calculate averages
        result = {}
        for cat, stats in category_stats.items():
            result[cat] = {
                "count": stats["count"],
                "avg_confidence": round(
                    sum(stats["confidences"]) / len(stats["confidences"]), 3
                ),
                "avg_conflicts": round(
                    sum(stats["conflict_counts"]) / len(stats["conflict_counts"]), 2
                ),
            }
        
        return result
    
    def _evaluate_rule_usage(self, decisions: List) -> Dict:
        """Analyze which rules are most frequently dominant."""
        dominant_counter = Counter()
        triggered_counter = Counter()
        violated_counter = Counter()
        
        for d in decisions:
            dominant_counter[d.dominant_rule] += 1
            for action_score in d.action_scores:
                for rule_id in action_score.triggered_rules:
                    triggered_counter[rule_id] += 1
                for rule_id in action_score.violated_rules:
                    violated_counter[rule_id] += 1
        
        return {
            "most_dominant_rules": dominant_counter.most_common(10),
            "most_triggered_rules": triggered_counter.most_common(10),
            "most_violated_rules": violated_counter.most_common(10),
        }
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate an overall moral reasoning score."""
        scores = []
        
        # Consistency (weight: 0.2)
        consistency = metrics.get("moral_consistency", {}).get("average_consistency", 0)
        scores.append(consistency * 0.2)
        
        # Harm minimization (weight: 0.25)
        harm = metrics.get("harm_minimization", {}).get("average_harm_minimization", 0)
        scores.append(harm * 0.25)
        
        # Fairness (weight: 0.2)
        fairness = metrics.get("fairness", {}).get("average_fairness", 0)
        scores.append(fairness * 0.2)
        
        # Transparency (weight: 0.15)
        transparency = metrics.get("transparency", {}).get("transparency_score", 0)
        scores.append(transparency * 0.15)
        
        # Confidence (weight: 0.1)
        confidence = metrics.get("confidence_distribution", {}).get("average_confidence", 0)
        scores.append(confidence * 0.1)
        
        # Inverse conflict rate (weight: 0.1) - fewer conflicts = better
        conflict_rate = metrics.get("conflict_analysis", {}).get("conflict_rate", 1)
        scores.append((1 - conflict_rate) * 0.1)
        
        return round(sum(scores), 3)
    
    def _interpret_consistency(self, score: float) -> str:
        if score > 0.8:
            return "Highly consistent - the model applies rules uniformly"
        elif score > 0.6:
            return "Moderately consistent - some variation in rule application"
        elif score > 0.4:
            return "Inconsistent - significant variation in decisions"
        else:
            return "Highly inconsistent - potential systematic issues"
    
    def _interpret_harm(self, score: float) -> str:
        if score > 0.7:
            return "Strong harm minimization - consistently chooses less harmful options"
        elif score > 0.5:
            return "Moderate harm minimization - usually reduces harm"
        else:
            return "Weak harm minimization - may not prioritize reducing harm"
    
    def _interpret_fairness(self, score: float) -> str:
        if score > 0.7:
            return "High fairness - decisions are equitable"
        elif score > 0.5:
            return "Moderate fairness - generally fair but with exceptions"
        else:
            return "Low fairness - potential bias in decisions"
    
    def export_results(self, filepath: str):
        """Export evaluation results to JSON."""
        # Convert Counter objects and tuples for JSON serialization
        serializable = []
        for result in self.results:
            sr = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    sr[key] = self._make_serializable(value)
                else:
                    sr[key] = value
            serializable.append(sr)
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
    
    def _make_serializable(self, obj):
        """Make an object JSON serializable."""
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
