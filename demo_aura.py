"""
Live evaluation demo against Neo4j Aura
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from neo4j_engine.connector import Neo4jConnector
from neo4j_engine.queries import EthicalGraphQueries
from neo4j_engine.reasoning import GraphReasoningEngine
from neo4j_engine.explanation import GraphExplanationGenerator
from core.rules import EthicalRuleSystem
from core.engine import MoralDecisionEngine
from data.scenarios import get_scenario_by_id, get_all_scenarios

URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASS = os.getenv("NEO4J_PASSWORD")
DB   = os.getenv("NEO4J_DATABASE")

conn = Neo4jConnector(uri=URI, user=USER, password=PASS, database=DB)
conn.connect()

queries = EthicalGraphQueries(conn)
engine = GraphReasoningEngine(queries)
explainer = GraphExplanationGenerator(queries)

# --- 1. Graph-only evaluation ---
print("=" * 60)
print("  1. GRAPH-ONLY EVALUATION: AV_01")
print("=" * 60)
decision = engine.evaluate_scenario("AV_01")
report = explainer.generate_text_explanation(decision.__dict__)
print(report)

# --- 2. Model 1 Hybrid ---
print("\n" + "=" * 60)
print("  2. MODEL 1 HYBRID EVALUATION: AV_01")
print("=" * 60)
scenario = get_scenario_by_id("AV_01")
rule_system = EthicalRuleSystem()
model1 = MoralDecisionEngine(rule_system)
hybrid_result = engine.evaluate_with_model1(scenario, model1, graph_weight=0.4)
print(explainer.generate_text_explanation(hybrid_result))

# --- 3. Virtue Conflicts ---
print("\n" + "=" * 60)
print("  3. ALL VIRTUE CONFLICTS")
print("=" * 60)
conflicts = queries.find_all_virtue_conflicts()
for c in conflicts:
    print(f"  {c['virtue_a']} <-> {c['virtue_b']}  tension={c['tension']}")

# --- 4. Multi-scenario demo ---
print("\n" + "=" * 60)
print("  4. GRAPH SCORES: FIRST 10 SCENARIOS")
print("=" * 60)
scenarios = get_all_scenarios()[:10]
for s in scenarios:
    try:
        d = engine.evaluate_scenario(s["id"])
        summary = explainer.summarize(d.__dict__)
        print(f"  {summary}")
    except Exception as e:
        print(f"  [{s['id']}] Error: {e}")

# --- 5. Structured explanation ---
print("\n" + "=" * 60)
print("  5. STRUCTURED EXPLANATION: AV_06")
print("=" * 60)
d6 = engine.evaluate_scenario("AV_06")
structured = explainer.generate_structured_explanation(d6.__dict__)
print(f"  Scenario: {structured['scenario_id']}")
print(f"  Chosen:   {structured['chosen_action']}")
print(f"  Score:    {structured['moral_score']}")
print(f"  Source:   {structured['reasoning_source']}")
print(f"  Virtues:  {structured['virtues_involved']}")
print(f"  Principles: {list(structured['principles_involved'].keys())}")
print(f"  Path len: {structured['path_length']}")

conn.close()
print("\nDone.")
