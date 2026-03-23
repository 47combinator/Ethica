"""
Model 1: Rule-Based Moral AI — CLI Runner
==========================================
Run the moral decision engine on the AMR-220 dataset.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rules import EthicalRuleSystem
from core.engine import MoralDecisionEngine
from core.explanation import ExplanationGenerator
from core.evaluation import EvaluationSystem
from data.scenarios import get_all_scenarios, get_scenarios_by_category, CATEGORY_NAMES, get_category_counts


def run_single_scenario(scenario_id=None):
    """Run a single scenario and print the explanation."""
    from data.scenarios import get_scenario_by_id
    
    rule_system = EthicalRuleSystem()
    engine = MoralDecisionEngine(rule_system)
    explainer = ExplanationGenerator(rule_system)
    
    if scenario_id:
        scenario = get_scenario_by_id(scenario_id)
        if not scenario:
            print(f"Scenario '{scenario_id}' not found.")
            return
    else:
        scenarios = get_all_scenarios()
        scenario = scenarios[0]
    
    decision = engine.evaluate_scenario(scenario)
    explanation = explainer.generate_full_explanation(decision, scenario)
    print(explanation)


def run_full_evaluation():
    """Run the full AMR-220 evaluation."""
    rule_system = EthicalRuleSystem()
    engine = MoralDecisionEngine(rule_system)
    evaluator = EvaluationSystem()
    explainer = ExplanationGenerator(rule_system)
    
    scenarios = get_all_scenarios()
    print(f"Running evaluation on {len(scenarios)} scenarios...")
    print(f"Categories: {len(CATEGORY_NAMES)}")
    print()
    
    # Run all scenarios
    decisions = engine.batch_evaluate(scenarios)
    
    # Evaluate
    metrics = evaluator.evaluate_decisions(decisions, scenarios)
    
    # Print results
    print("=" * 60)
    print("AMR-220 EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Scenarios: {metrics['total_scenarios']}")
    print(f"Overall Score: {metrics['overall_score']:.3f}")
    
    print(f"\n--- Moral Consistency ---")
    mc = metrics['moral_consistency']
    print(f"  Average: {mc['average_consistency']:.3f}")
    print(f"  Interpretation: {mc['interpretation']}")
    
    print(f"\n--- Harm Minimization ---")
    hm = metrics['harm_minimization']
    print(f"  Average: {hm['average_harm_minimization']:.3f}")
    print(f"  Scenarios minimized: {hm['scenarios_minimized_harm']}/{hm['total_evaluated']}")
    
    print(f"\n--- Fairness ---")
    f = metrics['fairness']
    print(f"  Average: {f['average_fairness']:.3f}")
    print(f"  High fairness: {f['high_fairness_decisions']}")
    
    print(f"\n--- Conflict Analysis ---")
    ca = metrics['conflict_analysis']
    print(f"  Total conflicts: {ca['total_conflicts']}")
    print(f"  Conflict rate: {ca['conflict_rate']:.1%}")
    
    print(f"\n--- Transparency ---")
    t = metrics['transparency']
    print(f"  Score: {t['transparency_score']:.3f}")
    print(f"  Avg reasoning steps: {t['average_reasoning_steps']}")
    
    print(f"\n--- Confidence ---")
    c = metrics['confidence_distribution']
    print(f"  Average: {c['average_confidence']:.3f}")
    print(f"  High/Med/Low: {c['high_confidence']}/{c['medium_confidence']}/{c['low_confidence']}")
    
    print(f"\n--- Category Performance ---")
    for cat, stats in metrics['category_analysis'].items():
        cat_name = CATEGORY_NAMES.get(cat, cat)
        print(f"  {cat_name}: {stats['count']} scenarios, "
              f"conf={stats['avg_confidence']:.3f}, conflicts={stats['avg_conflicts']:.1f}")
    
    # Export results
    results_path = os.path.join(os.path.dirname(__file__), "data", "results", "evaluation_results.json")
    evaluator.export_results(results_path)
    print(f"\nResults exported to: {results_path}")
    
    # Print sample explanations
    print("\n" + "=" * 60)
    print("SAMPLE DECISIONS (first 3)")
    print("=" * 60)
    for i, (decision, scenario) in enumerate(zip(decisions[:3], scenarios[:3])):
        brief = explainer.generate_brief_explanation(decision, scenario)
        print(f"\n[{scenario['id']}] {scenario['title']}")
        print(f"  {brief}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--scenario":
            scenario_id = sys.argv[2] if len(sys.argv) > 2 else None
            run_single_scenario(scenario_id)
        elif sys.argv[1] == "--count":
            counts = get_category_counts()
            total = sum(counts.values())
            print(f"AMR-220 Dataset: {total} scenarios")
            for cat, count in counts.items():
                print(f"  {CATEGORY_NAMES.get(cat, cat)}: {count}")
        else:
            print("Usage: python main.py [--scenario ID] [--count]")
    else:
        run_full_evaluation()
