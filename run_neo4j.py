"""
Neo4j Graph Engine — CLI Runner & Demo
=======================================
Usage:
    python run_neo4j.py --ingest          # Ingest AMR-220 into Neo4j
    python run_neo4j.py --evaluate AV_01  # Evaluate a scenario (graph-only)
    python run_neo4j.py --hybrid AV_01    # Evaluate with Model 1 hybrid
    python run_neo4j.py --stats           # Print graph statistics
    python run_neo4j.py --conflicts       # List all virtue conflicts
    python run_neo4j.py --demo            # Run full demo on first 5 scenarios
"""

import sys, os, argparse, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_connector(args):
    from neo4j_engine.connector import Neo4jConnector
    conn = Neo4jConnector(
        uri=args.uri,
        user=args.user,
        password=args.password,
    )
    conn.connect()
    return conn


def cmd_ingest(args):
    """Ingest AMR-220 dataset + principles + virtues into Neo4j."""
    from data.scenarios import get_all_scenarios
    from core.rules import EthicalRuleSystem
    from model4.virtues import VirtueSystem

    conn = get_connector(args)
    try:
        if args.clear:
            conn.clear_graph()
        conn.ingest_full_dataset(
            scenarios=get_all_scenarios(),
            rule_system=EthicalRuleSystem(),
            virtue_system=VirtueSystem(),
        )
        print("✓ Ingestion complete.")
    finally:
        conn.close()


def cmd_evaluate(args):
    """Graph-only evaluation of a scenario."""
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine
    from neo4j_engine.explanation import GraphExplanationGenerator

    conn = get_connector(args)
    try:
        queries = EthicalGraphQueries(conn)
        engine = GraphReasoningEngine(queries)
        explainer = GraphExplanationGenerator(queries)

        decision = engine.evaluate_scenario(args.scenario)
        report = explainer.generate_text_explanation(decision.__dict__)
        print(report)
    finally:
        conn.close()


def cmd_hybrid(args):
    """Model 1 + Graph hybrid evaluation."""
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine
    from neo4j_engine.explanation import GraphExplanationGenerator
    from core.rules import EthicalRuleSystem
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id

    scenario = get_scenario_by_id(args.scenario)
    if not scenario:
        print(f"Scenario '{args.scenario}' not found.")
        return

    conn = get_connector(args)
    try:
        queries = EthicalGraphQueries(conn)
        graph_engine = GraphReasoningEngine(queries)
        explainer = GraphExplanationGenerator(queries)

        rule_system = EthicalRuleSystem()
        model1_engine = MoralDecisionEngine(rule_system)

        result = graph_engine.evaluate_with_model1(
            scenario, model1_engine, graph_weight=args.graph_weight
        )
        report = explainer.generate_text_explanation(result)
        print(report)
    finally:
        conn.close()


def cmd_stats(args):
    """Print graph node counts."""
    from neo4j_engine.queries import EthicalGraphQueries

    conn = get_connector(args)
    try:
        queries = EthicalGraphQueries(conn)
        stats = queries.graph_stats()
        print("Neo4j Graph Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    finally:
        conn.close()


def cmd_conflicts(args):
    """List all virtue conflicts."""
    from neo4j_engine.queries import EthicalGraphQueries

    conn = get_connector(args)
    try:
        queries = EthicalGraphQueries(conn)
        conflicts = queries.find_all_virtue_conflicts()
        print(f"Virtue Conflicts ({len(conflicts)} pairs):")
        for c in conflicts:
            print(f"  ⚡ {c['virtue_a']} ↔ {c['virtue_b']}  tension={c['tension']}")
    finally:
        conn.close()


def cmd_demo(args):
    """Full demo: evaluate first N scenarios with graph + hybrid."""
    from data.scenarios import get_all_scenarios
    from neo4j_engine.queries import EthicalGraphQueries
    from neo4j_engine.reasoning import GraphReasoningEngine
    from neo4j_engine.explanation import GraphExplanationGenerator

    conn = get_connector(args)
    try:
        queries = EthicalGraphQueries(conn)
        engine = GraphReasoningEngine(queries)
        explainer = GraphExplanationGenerator(queries)

        scenarios = get_all_scenarios()[:args.count]
        for scenario in scenarios:
            decision = engine.evaluate_scenario(scenario["id"])
            summary = explainer.summarize(decision.__dict__)
            print(summary)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ethica Neo4j Graph Engine")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")

    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest AMR-220 into Neo4j")
    p_ingest.add_argument("--clear", action="store_true", help="Clear graph first")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Graph-only scenario evaluation")
    p_eval.add_argument("scenario", help="Scenario ID (e.g. AV_01)")

    # hybrid
    p_hybrid = sub.add_parser("hybrid", help="Model 1 + Graph hybrid evaluation")
    p_hybrid.add_argument("scenario", help="Scenario ID")
    p_hybrid.add_argument("--graph-weight", type=float, default=0.4)

    # stats
    sub.add_parser("stats", help="Print graph node counts")

    # conflicts
    sub.add_parser("conflicts", help="List virtue conflicts")

    # demo
    p_demo = sub.add_parser("demo", help="Run demo on first N scenarios")
    p_demo.add_argument("--count", type=int, default=5)

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "evaluate": cmd_evaluate,
        "hybrid": cmd_hybrid,
        "stats": cmd_stats,
        "conflicts": cmd_conflicts,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
