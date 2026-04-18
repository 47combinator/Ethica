"""
Batched Ingest for Neo4j Aura
==============================
Uses UNWIND-based batch writes for fast remote ingestion.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from data.scenarios import get_all_scenarios
from core.rules import EthicalRuleSystem
from model4.virtues import VirtueSystem
from neo4j_engine.schema import (
    GraphSchema, SCHEMA_CONSTRAINTS, SCHEMA_INDEXES,
    CONSEQUENCE_PRINCIPLE_MAP, PRINCIPLE_VIRTUE_MAP, VIRTUE_CONFLICTS,
)

URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASS = os.getenv("NEO4J_PASSWORD")
DB   = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
driver.verify_connectivity()
print(f"Connected to {URI}")


def run(query, params=None):
    with driver.session(database=DB) as s:
        s.run(query, params or {})


def run_read(query, params=None):
    with driver.session(database=DB) as s:
        return [r.data() for r in s.run(query, params or {})]


# 1. Clear
print("Clearing graph...")
run("MATCH (n) DETACH DELETE n")

# 2. Schema
print("Creating constraints & indexes...")
for c in SCHEMA_CONSTRAINTS:
    try: run(c)
    except: pass
for c in SCHEMA_INDEXES:
    try: run(c)
    except: pass

# 3. Principles (batch)
print("Inserting principles...")
rule_system = EthicalRuleSystem()
principle_data = []
for r in rule_system.get_all_rules_sorted():
    principle_data.append({
        "name": r.name, "weight": r.weight, "priority": r.priority,
        "category": r.category.value, "description": r.description,
    })
run("""
UNWIND $items AS item
MERGE (p:Principle {name: item.name})
SET p.weight = item.weight, p.priority = item.priority,
    p.category = item.category, p.description = item.description
""", {"items": principle_data})
print(f"  {len(principle_data)} principles")

# 4. Virtues (batch)
print("Inserting virtues...")
virtue_system = VirtueSystem()
virtue_data = []
for v in virtue_system.get_all_virtues():
    virtue_data.append({
        "name": v.name, "category": v.category.value,
        "base_weight": v.base_weight, "description": v.description,
    })
run("""
UNWIND $items AS item
MERGE (v:Virtue {name: item.name})
SET v.category = item.category, v.base_weight = item.base_weight,
    v.description = item.description
""", {"items": virtue_data})
print(f"  {len(virtue_data)} virtues")

# 5. Scenarios + Actions + Consequences (batch)
print("Inserting scenarios, actions, consequences...")
scenarios = get_all_scenarios()

# Batch scenarios
scenario_data = [
    {"id": s["id"], "category": s["category"],
     "title": s.get("title", s["id"]), "description": s.get("description", "")}
    for s in scenarios
]
run("""
UNWIND $items AS item
MERGE (s:Scenario {id: item.id})
SET s.category = item.category, s.title = item.title,
    s.description = item.description
""", {"items": scenario_data})
print(f"  {len(scenario_data)} scenarios")

# Batch actions
action_data = []
for s in scenarios:
    for a in s.get("actions", []):
        action_data.append({
            "id": f"{s['id']}_{a['id']}",
            "description": a["description"],
            "scenario_id": s["id"],
        })
run("""
UNWIND $items AS item
MERGE (a:Action {id: item.id})
SET a.description = item.description, a.scenario_id = item.scenario_id
""", {"items": action_data})
print(f"  {len(action_data)} actions")

# Batch Scenario->Action links
run("""
UNWIND $items AS item
MATCH (s:Scenario {id: item.scenario_id})
MATCH (a:Action {id: item.id})
MERGE (s)-[:HAS_ACTION]->(a)
""", {"items": action_data})
print("  Scenario->Action links created")

# Batch consequences + Action->Consequence links
consequence_data = []
for s in scenarios:
    for a in s.get("actions", []):
        aid = f"{s['id']}_{a['id']}"
        for ctype, value in a.get("consequences", {}).items():
            consequence_data.append({
                "action_id": aid,
                "type": ctype,
                "severity": round(min(1.0, max(0.0, value)), 3),
                "value": round(value, 3),
            })

# Split consequences into chunks of 500 for Aura free tier
CHUNK = 500
for i in range(0, len(consequence_data), CHUNK):
    chunk = consequence_data[i:i+CHUNK]
    run("""
    UNWIND $items AS item
    MATCH (a:Action {id: item.action_id})
    CREATE (c:Consequence {type: item.type, severity: item.severity, value: item.value})
    MERGE (a)-[:LEADS_TO]->(c)
    """, {"items": chunk})
    print(f"  consequences chunk {i//CHUNK+1}/{(len(consequence_data)-1)//CHUNK+1}")

print(f"  {len(consequence_data)} total consequences")

# 6. Consequence->Principle links (batch)
print("Linking Consequence->Principle...")
cp_links = []
for ctype, principles in CONSEQUENCE_PRINCIPLE_MAP.items():
    for pname, influence in principles:
        cp_links.append({
            "consequence_type": ctype,
            "principle_name": pname,
            "influence": influence,
        })
run("""
UNWIND $items AS item
MATCH (c:Consequence {type: item.consequence_type})
MATCH (p:Principle {name: item.principle_name})
MERGE (c)-[r:AFFECTS]->(p)
SET r.influence = item.influence
""", {"items": cp_links})
print(f"  {len(cp_links)} C->P link types")

# 7. Principle->Virtue links (batch)
print("Linking Principle->Virtue...")
pv_links = []
for pname, virtues in PRINCIPLE_VIRTUE_MAP.items():
    for vname, alignment in virtues:
        pv_links.append({
            "principle_name": pname,
            "virtue_name": vname,
            "alignment": alignment,
        })
run("""
UNWIND $items AS item
MATCH (p:Principle {name: item.principle_name})
MATCH (v:Virtue {name: item.virtue_name})
MERGE (p)-[r:RELATES_TO]->(v)
SET r.alignment = item.alignment
""", {"items": pv_links})
print(f"  {len(pv_links)} P->V links")

# 8. Virtue conflicts (batch)
print("Linking Virtue conflicts...")
conflict_data = [
    {"virtue_a": a, "virtue_b": b, "tension": t}
    for a, b, t in VIRTUE_CONFLICTS
]
run("""
UNWIND $items AS item
MATCH (v1:Virtue {name: item.virtue_a})
MATCH (v2:Virtue {name: item.virtue_b})
MERGE (v1)-[r:CONFLICTS_WITH]->(v2)
SET r.tension = item.tension
MERGE (v2)-[r2:CONFLICTS_WITH]->(v1)
SET r2.tension = item.tension
""", {"items": conflict_data})
print(f"  {len(conflict_data)} conflict pairs")

# 9. Stats
print("\nVerifying graph...")
stats = run_read("""
MATCH (s:Scenario) WITH count(s) AS scenarios
MATCH (a:Action)   WITH scenarios, count(a) AS actions
MATCH (c:Consequence) WITH scenarios, actions, count(c) AS consequences
MATCH (p:Principle) WITH scenarios, actions, consequences, count(p) AS principles
MATCH (v:Virtue)
RETURN scenarios, actions, consequences, principles, count(v) AS virtues
""")
if stats:
    print("Graph Statistics:")
    for k, v in stats[0].items():
        print(f"  {k}: {v}")

driver.close()
print("\nIngestion complete!")
