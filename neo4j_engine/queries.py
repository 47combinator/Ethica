"""
Cypher Query Library for Ethica
================================
Reusable, parameterized Cypher queries for ethical reasoning.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Query Constants ──────────────────────────────────────────────────────────

GET_ACTIONS_FOR_SCENARIO = """
MATCH (s:Scenario {id: $scenario_id})-[:HAS_ACTION]->(a:Action)
RETURN a.id          AS action_id,
       a.description AS description
ORDER BY a.id
"""

GET_CONSEQUENCES_FOR_ACTION = """
MATCH (a:Action {id: $action_id})-[:LEADS_TO]->(c:Consequence)
RETURN c.type     AS type,
       c.severity AS severity,
       c.value    AS value
ORDER BY c.severity DESC
"""

COMPUTE_MORAL_SCORE = """
MATCH (s:Scenario {id: $scenario_id})-[:HAS_ACTION]->(a:Action)
OPTIONAL MATCH (a)-[:LEADS_TO]->(c:Consequence)-[af:AFFECTS]->(p:Principle)
WITH a,
     collect({
         type:      c.type,
         severity:  c.severity,
         value:     c.value,
         principle: p.name,
         weight:    p.weight,
         influence: af.influence
     }) AS factors
WITH a,
     factors,
     CASE WHEN size(factors) > 0
          THEN reduce(s = 0.0, f IN factors |
               s + (1.0 - f.value) * f.weight * f.influence
          ) / size(factors)
          ELSE 0.0
     END AS moral_score
RETURN a.id          AS action_id,
       a.description AS description,
       round(moral_score, 4) AS moral_score,
       size(factors) AS factor_count
ORDER BY moral_score DESC
"""

GET_BEST_ACTION = """
MATCH (s:Scenario {id: $scenario_id})-[:HAS_ACTION]->(a:Action)
OPTIONAL MATCH (a)-[:LEADS_TO]->(c:Consequence)-[af:AFFECTS]->(p:Principle)
WITH a,
     collect({value: c.value, weight: p.weight, influence: af.influence}) AS factors
WITH a,
     CASE WHEN size(factors) > 0
          THEN reduce(s = 0.0, f IN factors |
               s + (1.0 - f.value) * f.weight * f.influence
          ) / size(factors)
          ELSE 0.0
     END AS moral_score
ORDER BY moral_score DESC
LIMIT 1
RETURN a.id          AS action_id,
       a.description AS description,
       round(moral_score, 4) AS moral_score
"""

GET_EXPLANATION_PATH = """
MATCH (s:Scenario {id: $scenario_id})-[:HAS_ACTION]->(a:Action {id: $action_id})
MATCH (a)-[:LEADS_TO]->(c:Consequence)-[af:AFFECTS]->(p:Principle)
OPTIONAL MATCH (p)-[rt:RELATES_TO]->(v:Virtue)
RETURN s.id          AS scenario_id,
       s.title       AS scenario_title,
       a.id          AS action_id,
       a.description AS action_desc,
       c.type        AS consequence_type,
       c.value       AS consequence_value,
       af.influence  AS influence,
       p.name        AS principle,
       p.weight      AS principle_weight,
       rt.alignment  AS alignment,
       v.name        AS virtue
ORDER BY af.influence DESC, rt.alignment DESC
"""

FIND_CONFLICTING_VIRTUES = """
MATCH (v1:Virtue)-[r:CONFLICTS_WITH]->(v2:Virtue)
WHERE v1.name < v2.name
RETURN v1.name   AS virtue_a,
       v2.name   AS virtue_b,
       r.tension AS tension
ORDER BY r.tension DESC
"""

FIND_CONFLICTING_VIRTUES_FOR_ACTION = """
MATCH (a:Action {id: $action_id})-[:LEADS_TO]->(c:Consequence)
      -[:AFFECTS]->(p:Principle)-[:RELATES_TO]->(v:Virtue)
WITH collect(DISTINCT v.name) AS involved_virtues
MATCH (v1:Virtue)-[r:CONFLICTS_WITH]->(v2:Virtue)
WHERE v1.name IN involved_virtues
  AND v2.name IN involved_virtues
  AND v1.name < v2.name
RETURN v1.name   AS virtue_a,
       v2.name   AS virtue_b,
       r.tension AS tension
ORDER BY r.tension DESC
"""

GRAPH_STATS = """
MATCH (s:Scenario) WITH count(s) AS scenarios
MATCH (a:Action)   WITH scenarios, count(a) AS actions
MATCH (c:Consequence) WITH scenarios, actions, count(c) AS consequences
MATCH (p:Principle) WITH scenarios, actions, consequences, count(p) AS principles
MATCH (v:Virtue)
RETURN scenarios, actions, consequences, principles, count(v) AS virtues
"""


class EthicalGraphQueries:
    """
    High-level query interface.

    Wraps raw Cypher queries and returns structured Python dicts.
    Requires a Neo4jConnector instance.
    """

    def __init__(self, connector):
        self.connector = connector

    def _run(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.connector._run(query, params)

    # ── Scenario queries ──────────────────────────────────────────────────

    def get_actions(self, scenario_id: str) -> List[Dict]:
        """Return all actions for a given scenario."""
        return self._run(GET_ACTIONS_FOR_SCENARIO, {"scenario_id": scenario_id})

    def get_consequences(self, action_id: str) -> List[Dict]:
        """Return all consequences for a given action."""
        return self._run(GET_CONSEQUENCES_FOR_ACTION, {"action_id": action_id})

    # ── Moral scoring ─────────────────────────────────────────────────────

    def compute_moral_scores(self, scenario_id: str) -> List[Dict]:
        """
        Compute a graph-based moral score for every action
        in the scenario using Consequence→Principle weights.
        """
        return self._run(COMPUTE_MORAL_SCORE, {"scenario_id": scenario_id})

    def get_best_action(self, scenario_id: str) -> Optional[Dict]:
        """Return the highest-scoring action for a scenario."""
        results = self._run(GET_BEST_ACTION, {"scenario_id": scenario_id})
        return results[0] if results else None

    # ── Explanation paths ─────────────────────────────────────────────────

    def get_explanation_path(self, scenario_id: str, action_id: str) -> List[Dict]:
        """
        Retrieve the full reasoning path:
        Scenario → Action → Consequence → Principle → Virtue
        """
        return self._run(
            GET_EXPLANATION_PATH,
            {"scenario_id": scenario_id, "action_id": action_id},
        )

    # ── Virtue conflicts ──────────────────────────────────────────────────

    def find_all_virtue_conflicts(self) -> List[Dict]:
        """Return all virtue conflict pairs in the graph."""
        return self._run(FIND_CONFLICTING_VIRTUES)

    def find_action_virtue_conflicts(self, action_id: str) -> List[Dict]:
        """Return virtue conflicts relevant to a specific action."""
        return self._run(
            FIND_CONFLICTING_VIRTUES_FOR_ACTION, {"action_id": action_id}
        )

    # ── Stats ─────────────────────────────────────────────────────────────

    def graph_stats(self) -> Dict:
        """Return node counts for each label."""
        results = self._run(GRAPH_STATS)
        return results[0] if results else {}
