"""
Offline Test Suite for Neo4j Ethical Reasoning Engine
=====================================================
Validates the full pipeline using mock data — no live Neo4j required.

Run:  python -m neo4j_engine.test_offline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j_engine.schema import GraphSchema, CONSEQUENCE_PRINCIPLE_MAP, PRINCIPLE_VIRTUE_MAP, VIRTUE_CONFLICTS
from neo4j_engine.explanation import GraphExplanationGenerator


# ─── Mock connector that returns canned query results ─────────────────────────

class MockConnector:
    """Simulates Neo4j responses for offline testing."""

    def __init__(self):
        self._data = self._build_mock_graph()

    def _build_mock_graph(self):
        """Build in-memory representation of a small test graph."""
        return {
            "scenarios": {
                "AV_01": {
                    "title": "Pedestrian vs Passenger Safety",
                    "category": "autonomous_vehicles",
                    "actions": {
                        "AV_01_A1": {
                            "description": "Continue forward (risk hitting pedestrian)",
                            "consequences": {
                                "harm_to_others": 0.9,
                                "harm_to_self": 0.1,
                                "lives_at_risk_score": 0.7,
                                "fairness_impact": 0.4,
                            },
                        },
                        "AV_01_A2": {
                            "description": "Swerve to avoid pedestrian (risk passenger injury)",
                            "consequences": {
                                "harm_to_others": 0.2,
                                "harm_to_self": 0.7,
                                "lives_at_risk_score": 0.5,
                                "fairness_impact": 0.6,
                            },
                        },
                    },
                },
            },
        }

    def _run(self, query, params=None):
        """Route queries to mock handlers based on content."""
        params = params or {}
        q = query.strip()

        if "HAS_ACTION" in q and "LEADS_TO" not in q:
            return self._mock_get_actions(params)
        if "LEADS_TO" in q and "AFFECTS" not in q:
            return self._mock_get_consequences(params)
        if "moral_score" in q and "LIMIT 1" in q:
            return self._mock_best_action(params)
        if "moral_score" in q:
            return self._mock_moral_scores(params)
        # Check CONFLICTS_WITH before explanation path (both share AFFECTS/RELATES_TO)
        if "CONFLICTS_WITH" in q and "action_id" not in str(params):
            return self._mock_all_conflicts()
        if "CONFLICTS_WITH" in q:
            return self._mock_action_conflicts(params)
        if "AFFECTS" in q and "RELATES_TO" in q:
            return self._mock_explanation_path(params)
        if "count" in q.lower():
            return [{"scenarios": 1, "actions": 2, "consequences": 8, "principles": 22, "virtues": 8}]
        return []

    # ── Mock query handlers ───────────────────────────────────────────────

    def _mock_get_actions(self, params):
        sid = params.get("scenario_id", "AV_01")
        scenario = self._data["scenarios"].get(sid, {})
        return [
            {"action_id": aid, "description": a["description"]}
            for aid, a in scenario.get("actions", {}).items()
        ]

    def _mock_get_consequences(self, params):
        aid = params.get("action_id", "")
        for s in self._data["scenarios"].values():
            if aid in s["actions"]:
                cons = s["actions"][aid]["consequences"]
                return [
                    {"type": k, "severity": v, "value": v}
                    for k, v in cons.items()
                ]
        return []

    def _mock_moral_scores(self, params):
        sid = params.get("scenario_id", "AV_01")
        scenario = self._data["scenarios"].get(sid, {})
        scores = []
        for aid, action in scenario.get("actions", {}).items():
            cons = action["consequences"]
            # Simplified moral score: avg of (1 - harm_values) weighted by principle weights
            cpm = CONSEQUENCE_PRINCIPLE_MAP
            total, count = 0.0, 0
            for ctype, value in cons.items():
                if ctype in cpm:
                    for pname, influence in cpm[ctype]:
                        total += (1.0 - value) * 0.8 * influence  # 0.8 = avg weight
                        count += 1
            moral_score = round(total / max(count, 1), 4)
            scores.append({
                "action_id": aid,
                "description": action["description"],
                "moral_score": moral_score,
                "factor_count": count,
            })
        scores.sort(key=lambda x: -x["moral_score"])
        return scores

    def _mock_best_action(self, params):
        scores = self._mock_moral_scores(params)
        return [scores[0]] if scores else []

    def _mock_explanation_path(self, params):
        sid = params.get("scenario_id", "AV_01")
        aid = params.get("action_id", "AV_01_A2")
        scenario = self._data["scenarios"].get(sid, {})
        action = scenario.get("actions", {}).get(aid, {})
        cons = action.get("consequences", {})

        path = []
        for ctype, value in cons.items():
            if ctype in CONSEQUENCE_PRINCIPLE_MAP:
                for pname, influence in CONSEQUENCE_PRINCIPLE_MAP[ctype]:
                    virtues = PRINCIPLE_VIRTUE_MAP.get(pname, [])
                    for vname, alignment in virtues:
                        path.append({
                            "scenario_id": sid,
                            "scenario_title": scenario.get("title", sid),
                            "action_id": aid,
                            "action_desc": action["description"],
                            "consequence_type": ctype,
                            "consequence_value": value,
                            "influence": influence,
                            "principle": pname,
                            "principle_weight": 0.8,
                            "alignment": alignment,
                            "virtue": vname,
                        })
        path.sort(key=lambda x: (-x["influence"], -x["alignment"]))
        return path

    def _mock_all_conflicts(self):
        return [
            {"virtue_a": a, "virtue_b": b, "tension": t}
            for a, b, t in VIRTUE_CONFLICTS
            if a < b
        ]

    def _mock_action_conflicts(self, params):
        # Return a subset of conflicts relevant to the action
        return [
            {"virtue_a": "Compassion", "virtue_b": "Prudence", "tension": 0.6},
            {"virtue_a": "Compassion", "virtue_b": "Justice", "tension": 0.5},
        ]


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_schema():
    print("-- Test: Schema --")
    schema = GraphSchema()
    assert len(schema.get_constraints()) == 4, "Expected 4 constraints"
    assert len(schema.get_indexes()) == 3, "Expected 3 indexes"
    assert len(schema.get_consequence_principle_map()) == 18, "Expected 18 C→P mappings"
    assert len(schema.get_principle_virtue_map()) == 19, "Expected 19 P→V mappings"
    assert len(schema.get_virtue_conflicts()) == 10, "Expected 10 conflict pairs"
    print("   [OK] Schema validated\n")


def test_queries():
    print("-- Test: Queries --")
    from neo4j_engine.queries import EthicalGraphQueries

    mock = MockConnector()
    queries = EthicalGraphQueries(mock)

    # Actions
    actions = queries.get_actions("AV_01")
    assert len(actions) == 2, f"Expected 2 actions, got {len(actions)}"
    print(f"   [OK] get_actions: {len(actions)} actions")

    # Consequences
    cons = queries.get_consequences("AV_01_A1")
    assert len(cons) == 4, f"Expected 4 consequences, got {len(cons)}"
    print(f"   [OK] get_consequences: {len(cons)} consequences")

    # Moral scores
    scores = queries.compute_moral_scores("AV_01")
    assert len(scores) == 2
    assert scores[0]["moral_score"] > scores[1]["moral_score"], "A2 should score higher"
    print(f"   [OK] compute_moral_scores: A2={scores[0]['moral_score']}, A1={scores[1]['moral_score']}")

    # Best action
    best = queries.get_best_action("AV_01")
    assert best["action_id"] == "AV_01_A2", f"Expected A2, got {best['action_id']}"
    print(f"   [OK] get_best_action: {best['action_id']} (swerve)")

    # Explanation path
    path = queries.get_explanation_path("AV_01", "AV_01_A2")
    assert len(path) > 0
    print(f"   [OK] get_explanation_path: {len(path)} path entries")

    # Conflicts
    conflicts = queries.find_all_virtue_conflicts()
    assert len(conflicts) > 0
    print(f"   [OK] find_all_virtue_conflicts: {len(conflicts)} pairs")

    # Stats
    stats = queries.graph_stats()
    assert stats["scenarios"] == 1
    print(f"   [OK] graph_stats: {stats}")
    print()


def test_reasoning():
    print("-- Test: Reasoning Engine --")
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine

    mock = MockConnector()
    queries = EthicalGraphQueries(mock)
    engine = GraphReasoningEngine(queries)

    decision = engine.evaluate_scenario("AV_01")
    assert decision.scenario_id == "AV_01"
    assert decision.chosen_action_id == "AV_01_A2"
    assert decision.moral_score > 0
    assert len(decision.explanation_path) > 0
    assert decision.reasoning_source == "graph"
    print(f"   [OK] evaluate_scenario:")
    print(f"     Chosen: {decision.chosen_action_desc}")
    print(f"     Score:  {decision.moral_score}")
    print(f"     Path entries: {len(decision.explanation_path)}")
    print(f"     Conflicts: {len(decision.virtue_conflicts)}")
    print()


def test_explanation():
    print("-- Test: Explanation Generator --")
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine
    from neo4j_engine.explanation import GraphExplanationGenerator

    mock = MockConnector()
    queries = EthicalGraphQueries(mock)
    engine = GraphReasoningEngine(queries)
    explainer = GraphExplanationGenerator(queries)

    decision = engine.evaluate_scenario("AV_01")

    # Text explanation
    text = explainer.generate_text_explanation(decision.__dict__)
    assert "ETHICAL REASONING REPORT" in text
    assert "AV_01" in text
    assert "Reasoning Path" in text
    print(f"   [OK] generate_text_explanation: {len(text)} chars")

    # Structured explanation
    structured = explainer.generate_structured_explanation(decision.__dict__)
    assert "principles_involved" in structured
    assert "virtues_involved" in structured
    assert len(structured["virtues_involved"]) > 0
    print(f"   [OK] generate_structured_explanation:")
    print(f"     Principles: {list(structured['principles_involved'].keys())[:4]}...")
    print(f"     Virtues: {structured['virtues_involved']}")

    # Summary
    summary = explainer.summarize(decision.__dict__)
    assert "AV_01" in summary
    print(f"   [OK] summarize: {summary}")
    print()

    # Print full report
    print("-- Full Explanation Report --")
    print(text)


def test_model1_hybrid():
    print("-- Test: Model 1 Hybrid --")
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine
    from neo4j_engine.explanation import GraphExplanationGenerator
    from core.rules import EthicalRuleSystem
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id

    mock = MockConnector()
    queries = EthicalGraphQueries(mock)
    engine = GraphReasoningEngine(queries)
    explainer = GraphExplanationGenerator(queries)

    scenario = get_scenario_by_id("AV_01")
    rule_system = EthicalRuleSystem()
    model1 = MoralDecisionEngine(rule_system)

    result = engine.evaluate_with_model1(scenario, model1, graph_weight=0.4)
    assert result["reasoning_source"] == "model1_hybrid"
    assert len(result["all_scores"]) == 2

    best = result["chosen_action"]
    print(f"   [OK] evaluate_with_model1:")
    print(f"     Chosen: {best['description']}")
    print(f"     Blended: {best['blended_score']}  (M1={best['model1_score']}, Graph={best['graph_score']})")
    print(f"     Model 1 dominant rule: {result['model1_decision']['dominant_rule']}")

    report = explainer.generate_text_explanation(result)
    print()
    print("-- Hybrid Report --")
    print(report)


# ─── Run all ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ethica Neo4j Engine - Offline Test Suite")
    print("=" * 60)
    print()

    tests = [test_schema, test_queries, test_reasoning, test_explanation, test_model1_hybrid]
    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   [FAIL] FAILED: {e}\n")

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
