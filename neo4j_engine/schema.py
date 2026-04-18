"""
Neo4j Graph Schema for Ethica
=============================
Defines node types, relationships, and constraints for the
ethical reasoning knowledge graph.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Cypher: Schema Constraints & Indexes ────────────────────────────────────

SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT scenario_id IF NOT EXISTS FOR (s:Scenario) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT principle_name IF NOT EXISTS FOR (p:Principle) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT virtue_name IF NOT EXISTS FOR (v:Virtue) REQUIRE v.name IS UNIQUE",
]

SCHEMA_INDEXES = [
    "CREATE INDEX scenario_category IF NOT EXISTS FOR (s:Scenario) ON (s.category)",
    "CREATE INDEX consequence_type IF NOT EXISTS FOR (c:Consequence) ON (c.type)",
    "CREATE INDEX consequence_severity IF NOT EXISTS FOR (c:Consequence) ON (c.severity)",
]


# ─── Cypher: Node Creation Templates ─────────────────────────────────────────

CREATE_SCENARIO = """
MERGE (s:Scenario {id: $id})
SET s.category    = $category,
    s.title       = $title,
    s.description = $description
RETURN s
"""

CREATE_ACTION = """
MERGE (a:Action {id: $id})
SET a.description = $description,
    a.scenario_id = $scenario_id
RETURN a
"""

CREATE_CONSEQUENCE = """
CREATE (c:Consequence {
    type:     $type,
    severity: $severity,
    value:    $value
})
RETURN c
"""

CREATE_PRINCIPLE = """
MERGE (p:Principle {name: $name})
SET p.weight      = $weight,
    p.priority    = $priority,
    p.category    = $category,
    p.description = $description
RETURN p
"""

CREATE_VIRTUE = """
MERGE (v:Virtue {name: $name})
SET v.category    = $category,
    v.base_weight = $base_weight,
    v.description = $description
RETURN v
"""


# ─── Cypher: Relationship Creation Templates ─────────────────────────────────

LINK_SCENARIO_ACTION = """
MATCH (s:Scenario {id: $scenario_id})
MATCH (a:Action {id: $action_id})
MERGE (s)-[:HAS_ACTION]->(a)
"""

LINK_ACTION_CONSEQUENCE = """
MATCH (a:Action {id: $action_id})
CREATE (c:Consequence {type: $type, severity: $severity, value: $value})
MERGE (a)-[:LEADS_TO]->(c)
RETURN c
"""

LINK_CONSEQUENCE_PRINCIPLE = """
MATCH (c:Consequence)
WHERE c.type = $consequence_type AND c.severity >= $min_severity
MATCH (p:Principle {name: $principle_name})
MERGE (c)-[r:AFFECTS]->(p)
SET r.influence = $influence
"""

LINK_PRINCIPLE_VIRTUE = """
MATCH (p:Principle {name: $principle_name})
MATCH (v:Virtue {name: $virtue_name})
MERGE (p)-[r:RELATES_TO]->(v)
SET r.alignment = $alignment
"""

LINK_VIRTUE_CONFLICT = """
MATCH (v1:Virtue {name: $virtue_a})
MATCH (v2:Virtue {name: $virtue_b})
MERGE (v1)-[r:CONFLICTS_WITH]->(v2)
SET r.tension = $tension
MERGE (v2)-[r2:CONFLICTS_WITH]->(v1)
SET r2.tension = $tension
"""


# ─── Principle ↔ Consequence Mappings ─────────────────────────────────────────
# Maps consequence types to the principles they affect

CONSEQUENCE_PRINCIPLE_MAP = {
    "harm_to_others":           [("Preserve Human Life", 0.9), ("Minimize Direct Harm", 0.95)],
    "harm_to_self":             [("Minimize Direct Harm", 0.7)],
    "lives_at_risk_score":      [("Preserve Human Life", 1.0), ("Maximize Lives Saved", 0.95)],
    "fairness_impact":          [("Equal Treatment", 0.8), ("Non-Discrimination", 0.75)],
    "discrimination_level":     [("Non-Discrimination", 0.9), ("Equal Treatment", 0.8)],
    "collateral_damage":        [("Minimize Collateral Damage", 0.85)],
    "safety_risk":              [("Precautionary Principle", 0.8)],
    "accountability_score":     [("Accountability", 0.8)],
    "deception_level":          [("Truthfulness", 0.85), ("Transparency", 0.7)],
    "transparency_score":       [("Transparency", 0.8)],
    "privacy_impact":           [("Protect Privacy", 0.85)],
    "autonomy_impact":          [("Respect Individual Autonomy", 0.8)],
    "consent_violation":        [("Informed Consent", 0.85)],
    "proportionality_score":    [("Proportional Response", 0.8)],
    "benefit_score":            [("Maximize Overall Benefit", 0.75)],
    "welfare_impact":           [("Long-term Welfare", 0.8)],
    "human_oversight_maintained": [("Human Control Priority", 0.85)],
    "legal_violation_score":    [("Precautionary Principle", 0.6)],
}

# Maps principles to the virtues they relate to
PRINCIPLE_VIRTUE_MAP = {
    "Preserve Human Life":        [("Compassion", 0.9), ("Responsibility", 0.8)],
    "Maximize Lives Saved":       [("Compassion", 0.85), ("Justice", 0.7)],
    "Minimize Direct Harm":       [("Compassion", 0.9), ("Prudence", 0.7)],
    "Minimize Collateral Damage": [("Compassion", 0.8), ("Justice", 0.75)],
    "Equal Treatment":            [("Justice", 0.95)],
    "Non-Discrimination":         [("Justice", 0.9)],
    "Procedural Fairness":        [("Justice", 0.85), ("Honesty", 0.6)],
    "Precautionary Principle":    [("Prudence", 0.9), ("Responsibility", 0.8)],
    "Accountability":             [("Responsibility", 0.95)],
    "Respect Individual Autonomy": [("Justice", 0.7), ("Courage", 0.5)],
    "Informed Consent":           [("Honesty", 0.8), ("Justice", 0.7)],
    "Truthfulness":               [("Honesty", 0.95)],
    "Transparency":               [("Honesty", 0.9)],
    "Protect Privacy":            [("Justice", 0.7), ("Prudence", 0.6)],
    "Human Control Priority":     [("Prudence", 0.8), ("Responsibility", 0.75)],
    "Proportional Response":      [("Temperance", 0.85), ("Prudence", 0.8)],
    "Least Restrictive Means":    [("Temperance", 0.8), ("Justice", 0.65)],
    "Maximize Overall Benefit":   [("Benevolence", 0.9), ("Compassion", 0.7)],
    "Long-term Welfare":          [("Benevolence", 0.85), ("Responsibility", 0.8)],
}

# Virtue conflict pairs (from model4 opposing_virtues)
VIRTUE_CONFLICTS = [
    ("Compassion", "Prudence",      0.6),
    ("Compassion", "Justice",       0.5),
    ("Justice",    "Compassion",    0.5),
    ("Justice",    "Courage",       0.4),
    ("Honesty",    "Compassion",    0.5),
    ("Honesty",    "Prudence",      0.4),
    ("Courage",    "Prudence",      0.7),
    ("Courage",    "Responsibility", 0.5),
    ("Benevolence", "Justice",      0.4),
    ("Benevolence", "Prudence",     0.45),
]


@dataclass
class GraphSchema:
    """Manages the Neo4j schema lifecycle."""

    def get_constraints(self) -> List[str]:
        return SCHEMA_CONSTRAINTS

    def get_indexes(self) -> List[str]:
        return SCHEMA_INDEXES

    def get_consequence_principle_map(self) -> Dict:
        return CONSEQUENCE_PRINCIPLE_MAP

    def get_principle_virtue_map(self) -> Dict:
        return PRINCIPLE_VIRTUE_MAP

    def get_virtue_conflicts(self) -> List[tuple]:
        return VIRTUE_CONFLICTS
