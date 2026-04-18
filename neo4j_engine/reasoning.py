"""
Graph-Based Reasoning Engine for Ethica
========================================
Orchestrates ethical reasoning using Neo4j graph traversals.
Designed to integrate with Model 1 (rule-based) and Model 4 (virtue ethics).
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphDecision:
    """A moral decision produced by graph-based reasoning."""
    scenario_id: str
    chosen_action_id: str
    chosen_action_desc: str
    moral_score: float
    all_scores: List[Dict]
    explanation_path: List[Dict]
    virtue_conflicts: List[Dict]
    reasoning_source: str  # "graph", "model1", "model4", "ensemble"


class GraphReasoningEngine:
    """
    Ethical reasoning engine backed by Neo4j.

    Supports three modes:
     - standalone   : pure graph-based scoring
     - model1_hybrid: combines graph scores with Model 1 rule scores
     - model4_hybrid: combines graph scores with Model 4 virtue scores
    """

    def __init__(self, queries):
        """
        Args:
            queries: EthicalGraphQueries instance (already connected).
        """
        self.queries = queries

    # ── Pure graph reasoning ──────────────────────────────────────────────

    def evaluate_scenario(self, scenario_id: str) -> GraphDecision:
        """
        Evaluate a scenario using only graph-based moral scoring.
        """
        # 1. Score all actions
        scores = self.queries.compute_moral_scores(scenario_id)
        if not scores:
            raise ValueError(f"No actions found for scenario '{scenario_id}'")

        best = scores[0]  # Already ordered DESC by moral_score

        # 2. Get explanation path for chosen action
        path = self.queries.get_explanation_path(
            scenario_id, best["action_id"]
        )

        # 3. Get virtue conflicts for chosen action
        conflicts = self.queries.find_action_virtue_conflicts(best["action_id"])

        return GraphDecision(
            scenario_id=scenario_id,
            chosen_action_id=best["action_id"],
            chosen_action_desc=best["description"],
            moral_score=best["moral_score"],
            all_scores=scores,
            explanation_path=path,
            virtue_conflicts=conflicts,
            reasoning_source="graph",
        )

    # ── Model 1 hybrid ───────────────────────────────────────────────────

    def evaluate_with_model1(
        self,
        scenario: Dict,
        model1_engine,
        graph_weight: float = 0.4,
    ) -> Dict:
        """
        Combine graph moral scores with Model 1 rule-based scores.

        Args:
            scenario:      Full scenario dict (AMR-220 format).
            model1_engine: core.engine.MoralDecisionEngine instance.
            graph_weight:  0-1 blend factor (0 = pure Model 1, 1 = pure graph).

        Returns:
            Combined decision dict with both sources.
        """
        scenario_id = scenario["id"]

        # Model 1 decision
        m1_decision = model1_engine.evaluate_scenario(scenario)

        # Graph scores
        graph_scores = self.queries.compute_moral_scores(scenario_id)
        graph_map = {s["action_id"]: s["moral_score"] for s in graph_scores}

        # Blend scores
        combined = []
        m1_weight = 1.0 - graph_weight
        for action_score in m1_decision.action_scores:
            graph_action_id = f"{scenario_id}_{action_score.action_id}"
            g_score = graph_map.get(graph_action_id, 0.0)
            # Normalize model1 score to 0-1
            m1_norm = (action_score.total_score + 1.0) / 2.0
            blended = m1_weight * m1_norm + graph_weight * g_score
            combined.append({
                "action_id": action_score.action_id,
                "description": action_score.action_description,
                "model1_score": round(action_score.total_score, 4),
                "graph_score": round(g_score, 4),
                "blended_score": round(blended, 4),
            })

        combined.sort(key=lambda x: x["blended_score"], reverse=True)
        best = combined[0]

        # Explanation path for best action
        graph_action_id = f"{scenario_id}_{best['action_id']}"
        path = self.queries.get_explanation_path(scenario_id, graph_action_id)

        return {
            "scenario_id": scenario_id,
            "reasoning_source": "model1_hybrid",
            "graph_weight": graph_weight,
            "chosen_action": best,
            "all_scores": combined,
            "model1_decision": {
                "action": m1_decision.chosen_action,
                "confidence": m1_decision.confidence,
                "dominant_rule": m1_decision.dominant_rule,
            },
            "explanation_path": path,
        }

    # ── Model 4 hybrid ───────────────────────────────────────────────────

    def evaluate_with_model4(
        self,
        scenario: Dict,
        model4_evaluator,
        graph_weight: float = 0.4,
    ) -> Dict:
        """
        Combine graph moral scores with Model 4 virtue scores.

        Args:
            scenario:         Full scenario dict.
            model4_evaluator: model4.evaluator.VirtueEvaluator instance.
            graph_weight:     0-1 blend factor.

        Returns:
            Combined decision dict.
        """
        scenario_id = scenario["id"]

        # Model 4 decision
        m4_decision = model4_evaluator.evaluate_scenario(scenario)

        # Graph scores
        graph_scores = self.queries.compute_moral_scores(scenario_id)
        graph_map = {s["action_id"]: s["moral_score"] for s in graph_scores}

        # Blend scores
        combined = []
        m4_weight = 1.0 - graph_weight
        for action_result in m4_decision.get("action_scores", []):
            aid = action_result.get("action_id", "")
            graph_action_id = f"{scenario_id}_{aid}"
            g_score = graph_map.get(graph_action_id, 0.0)
            m4_score = action_result.get("virtue_score", 0.5)
            blended = m4_weight * m4_score + graph_weight * g_score
            combined.append({
                "action_id": aid,
                "description": action_result.get("description", ""),
                "model4_score": round(m4_score, 4),
                "graph_score": round(g_score, 4),
                "blended_score": round(blended, 4),
            })

        combined.sort(key=lambda x: x["blended_score"], reverse=True)
        best = combined[0] if combined else {}

        graph_action_id = f"{scenario_id}_{best.get('action_id', '')}"
        path = self.queries.get_explanation_path(scenario_id, graph_action_id)
        conflicts = self.queries.find_action_virtue_conflicts(graph_action_id)

        return {
            "scenario_id": scenario_id,
            "reasoning_source": "model4_hybrid",
            "graph_weight": graph_weight,
            "chosen_action": best,
            "all_scores": combined,
            "virtue_conflicts": conflicts,
            "explanation_path": path,
        }

    # ── Batch evaluation ──────────────────────────────────────────────────

    def batch_evaluate(self, scenario_ids: List[str]) -> List[GraphDecision]:
        """Evaluate multiple scenarios using pure graph reasoning."""
        return [self.evaluate_scenario(sid) for sid in scenario_ids]
