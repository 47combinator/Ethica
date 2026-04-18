"""
Neo4j Connector for Ethica
===========================
Handles database connection lifecycle, session management,
schema initialization, and bulk data ingestion from AMR-220.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None  # Graceful stub when driver not installed

from .schema import (
    GraphSchema,
    SCHEMA_CONSTRAINTS, SCHEMA_INDEXES,
    CREATE_SCENARIO, CREATE_ACTION,
    LINK_SCENARIO_ACTION, LINK_ACTION_CONSEQUENCE,
    CREATE_PRINCIPLE, CREATE_VIRTUE,
    LINK_CONSEQUENCE_PRINCIPLE, LINK_PRINCIPLE_VIRTUE,
    LINK_VIRTUE_CONFLICT,
)

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """
    Manages the Neo4j connection and provides high-level
    methods for populating the ethical knowledge graph.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        if GraphDatabase is None:
            raise ImportError(
                "neo4j driver not installed. Run: pip install neo4j"
            )
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    # ── Connection lifecycle ──────────────────────────────────────────────

    def connect(self):
        """Establish connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", self.uri)

    def close(self):
        """Close the driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Low-level helpers ─────────────────────────────────────────────────

    def _run(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a single Cypher query and return records as dicts."""
        with self._driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def _run_write(self, query: str, params: Optional[Dict] = None):
        """Execute a write transaction."""
        with self._driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(query, params or {}))

    # ── Schema bootstrap ──────────────────────────────────────────────────

    def initialize_schema(self):
        """Create constraints and indexes."""
        schema = GraphSchema()
        for cypher in schema.get_constraints():
            try:
                self._run(cypher)
            except Exception as e:
                logger.debug("Constraint may already exist: %s", e)
        for cypher in schema.get_indexes():
            try:
                self._run(cypher)
            except Exception as e:
                logger.debug("Index may already exist: %s", e)
        logger.info("Schema initialized.")

    def clear_graph(self):
        """Delete all nodes and relationships (use with caution)."""
        self._run("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared.")

    # ── Data ingestion ────────────────────────────────────────────────────

    def insert_scenario(self, scenario: Dict):
        """
        Insert a full AMR-220 scenario into the graph.

        Creates Scenario, Action, and Consequence nodes with
        all corresponding relationships.
        """
        sid = scenario["id"]

        # 1. Create Scenario node
        self._run_write(
            CREATE_SCENARIO,
            {
                "id": sid,
                "category": scenario["category"],
                "title": scenario.get("title", sid),
                "description": scenario.get("description", ""),
            },
        )

        # 2. Create Action + Consequence nodes for each action
        for action in scenario.get("actions", []):
            action_id = f"{sid}_{action['id']}"

            self._run_write(
                CREATE_ACTION,
                {
                    "id": action_id,
                    "description": action["description"],
                    "scenario_id": sid,
                },
            )

            # Link Scenario → Action
            self._run_write(
                LINK_SCENARIO_ACTION,
                {"scenario_id": sid, "action_id": action_id},
            )

            # Create Consequence nodes from consequence dict
            consequences = action.get("consequences", {})
            for ctype, value in consequences.items():
                severity = self._value_to_severity(value)
                self._run_write(
                    LINK_ACTION_CONSEQUENCE,
                    {
                        "action_id": action_id,
                        "type": ctype,
                        "severity": severity,
                        "value": round(value, 3),
                    },
                )

    def insert_principles(self, rule_system):
        """
        Populate Principle nodes from Model 1's EthicalRuleSystem.
        """
        for rule in rule_system.get_all_rules_sorted():
            self._run_write(
                CREATE_PRINCIPLE,
                {
                    "name": rule.name,
                    "weight": rule.weight,
                    "priority": rule.priority,
                    "category": rule.category.value,
                    "description": rule.description,
                },
            )
        logger.info("Principles ingested from rule system.")

    def insert_virtues(self, virtue_system):
        """
        Populate Virtue nodes from Model 4's VirtueSystem.
        """
        for virtue in virtue_system.get_all_virtues():
            self._run_write(
                CREATE_VIRTUE,
                {
                    "name": virtue.name,
                    "category": virtue.category.value,
                    "base_weight": virtue.base_weight,
                    "description": virtue.description,
                },
            )
        logger.info("Virtues ingested from virtue system.")

    def link_consequences_to_principles(self):
        """
        Create AFFECTS edges between Consequence and Principle nodes
        based on the mapping in schema.py.
        """
        schema = GraphSchema()
        for ctype, principles in schema.get_consequence_principle_map().items():
            for principle_name, influence in principles:
                self._run_write(
                    LINK_CONSEQUENCE_PRINCIPLE,
                    {
                        "consequence_type": ctype,
                        "min_severity": 0.0,
                        "principle_name": principle_name,
                        "influence": influence,
                    },
                )
        logger.info("Consequence → Principle links created.")

    def link_principles_to_virtues(self):
        """
        Create RELATES_TO edges between Principle and Virtue nodes.
        """
        schema = GraphSchema()
        for principle_name, virtues in schema.get_principle_virtue_map().items():
            for virtue_name, alignment in virtues:
                self._run_write(
                    LINK_PRINCIPLE_VIRTUE,
                    {
                        "principle_name": principle_name,
                        "virtue_name": virtue_name,
                        "alignment": alignment,
                    },
                )
        logger.info("Principle → Virtue links created.")

    def link_virtue_conflicts(self):
        """
        Create CONFLICTS_WITH edges between opposing Virtue nodes.
        """
        schema = GraphSchema()
        for v_a, v_b, tension in schema.get_virtue_conflicts():
            self._run_write(
                LINK_VIRTUE_CONFLICT,
                {"virtue_a": v_a, "virtue_b": v_b, "tension": tension},
            )
        logger.info("Virtue conflict links created.")

    def ingest_full_dataset(self, scenarios: List[Dict], rule_system, virtue_system):
        """
        One-shot ingestion: schema → principles → virtues → scenarios → links.
        """
        self.initialize_schema()
        self.insert_principles(rule_system)
        self.insert_virtues(virtue_system)
        for scenario in scenarios:
            self.insert_scenario(scenario)
        self.link_consequences_to_principles()
        self.link_principles_to_virtues()
        self.link_virtue_conflicts()
        logger.info(
            "Full dataset ingested: %d scenarios.", len(scenarios)
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _value_to_severity(value: float) -> float:
        """Normalize a 0-1 consequence value to a severity score."""
        return round(min(1.0, max(0.0, value)), 3)
