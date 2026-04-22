"""
AMR Dataset Validator & Merger
===============================
Validates all converted scenarios and merges them with the existing
AMR-220 dataset to produce AMR-1000.

Checks:
  1. Unique IDs
  2. Valid categories
  3. 2-3 actions per scenario
  4. Consequence values in [0.0, 1.0]
  5. No trivially dominant actions
  6. Minimum ethical dimensions
  7. Value distribution health
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_CATEGORIES = [
    "autonomous_vehicles", "healthcare_ai", "hiring_bias",
    "military_ai", "privacy_surveillance", "financial_ai",
    "disaster_response", "human_ai_interaction",
    "corporate_pressure", "moral_ambiguity",
    # New categories
    "education_ai", "judicial_ai", "environmental_ai",
    "creative_ai", "social_media_ai",
]

VALID_DIMENSIONS = [
    "harm", "life_preservation", "fairness", "autonomy", "honesty",
    "privacy", "responsibility", "human_oversight", "proportionality",
    "beneficence", "discrimination", "deception", "legal_compliance",
    "transparency", "consent", "manipulation", "safety", "welfare",
]


def validate_scenario(scenario: Dict) -> Tuple[List[str], List[str]]:
    """Validate a single scenario. Returns (errors, warnings)."""
    errors = []
    warnings = []
    sid = scenario.get("id", "UNKNOWN")

    # Basic structure
    if "id" not in scenario:
        errors.append(f"Missing 'id'")
    if "category" not in scenario:
        errors.append(f"{sid}: Missing 'category'")
    elif scenario["category"] not in VALID_CATEGORIES:
        warnings.append(f"{sid}: Unknown category '{scenario['category']}'")

    if "title" not in scenario:
        errors.append(f"{sid}: Missing 'title'")
    if "description" not in scenario:
        errors.append(f"{sid}: Missing 'description'")

    # Actions
    actions = scenario.get("actions", [])
    if len(actions) < 2:
        errors.append(f"{sid}: Only {len(actions)} action(s), need >= 2")
    elif len(actions) > 3:
        warnings.append(f"{sid}: {len(actions)} actions (usually max 3)")

    # Consequences
    for a in actions:
        if "id" not in a:
            errors.append(f"{sid}: Action missing 'id'")
        if "description" not in a:
            errors.append(f"{sid}/{a.get('id','?')}: Missing description")

        cons = a.get("consequences", {})
        if len(cons) < 4:
            warnings.append(f"{sid}/{a.get('id','?')}: Only {len(cons)} consequence keys")

        for k, v in cons.items():
            if not isinstance(v, (int, float)):
                errors.append(f"{sid}/{a.get('id','?')}: {k} is not numeric ({type(v).__name__})")
            elif v < 0.0 or v > 1.0:
                errors.append(f"{sid}/{a.get('id','?')}: {k}={v} out of [0, 1] range")

    # Dimensions
    dims = scenario.get("ethical_dimensions", [])
    if len(dims) < 2:
        warnings.append(f"{sid}: Only {len(dims)} ethical dimensions (want >= 3)")
    for d in dims:
        if d not in VALID_DIMENSIONS:
            warnings.append(f"{sid}: Unknown dimension '{d}'")

    # Dominance check
    if len(actions) >= 2:
        for i, a1 in enumerate(actions):
            for j, a2 in enumerate(actions):
                if i >= j:
                    continue
                c1 = a1.get("consequences", {})
                c2 = a2.get("consequences", {})
                shared = set(c1.keys()) & set(c2.keys())
                if len(shared) >= 4:
                    # Check if a1 is strictly better (lower harm, higher benefit)
                    better_count = 0
                    for k in shared:
                        harm_keys = ["harm_to_others", "harm_to_self", "lives_at_risk_score",
                                     "safety_risk", "collateral_damage", "discrimination_level",
                                     "deception_level", "privacy_impact", "consent_violation",
                                     "manipulation_level", "legal_violation_score", "data_exposure",
                                     "restrictiveness"]
                        if k in harm_keys:
                            if c1[k] <= c2[k]:
                                better_count += 1
                        else:
                            if c1[k] >= c2[k]:
                                better_count += 1

                    dom_ratio = better_count / len(shared)
                    if dom_ratio > 0.85:
                        warnings.append(
                            f"{sid}: {a1.get('id','?')} dominates {a2.get('id','?')} "
                            f"({dom_ratio:.0%} of shared keys)"
                        )

    return errors, warnings


def validate_dataset(scenarios: List[Dict]) -> Dict:
    """Validate entire dataset. Returns summary report."""
    all_errors = []
    all_warnings = []
    ids = set()
    duplicate_ids = []

    for s in scenarios:
        sid = s.get("id", "UNKNOWN")
        if sid in ids:
            duplicate_ids.append(sid)
        ids.add(sid)

        errs, warns = validate_scenario(s)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    if duplicate_ids:
        all_errors.extend([f"DUPLICATE ID: {d}" for d in duplicate_ids])

    # Distribution analysis
    cats = Counter(s.get("category", "?") for s in scenarios)
    dims = Counter()
    for s in scenarios:
        for d in s.get("ethical_dimensions", []):
            dims[d] += 1

    action_counts = Counter(len(s.get("actions", [])) for s in scenarios)
    three_action_pct = action_counts.get(3, 0) / max(len(scenarios), 1) * 100

    # Value distribution
    all_values = []
    for s in scenarios:
        for a in s.get("actions", []):
            for v in a.get("consequences", {}).values():
                if isinstance(v, (int, float)):
                    all_values.append(v)

    if all_values:
        import statistics
        val_mean = statistics.mean(all_values)
        val_stdev = statistics.stdev(all_values)
        val_median = statistics.median(all_values)
    else:
        val_mean = val_stdev = val_median = 0

    # Missing dimensions
    missing_dims = [d for d in VALID_DIMENSIONS if dims.get(d, 0) < 5]

    report = {
        "total_scenarios": len(scenarios),
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
        "errors": all_errors[:30],
        "warnings": all_warnings[:30],
        "category_distribution": dict(cats.most_common()),
        "dimension_distribution": dict(dims.most_common()),
        "action_distribution": dict(action_counts),
        "three_action_percentage": round(three_action_pct, 1),
        "value_stats": {
            "mean": round(val_mean, 3),
            "median": round(val_median, 3),
            "stdev": round(val_stdev, 3),
            "total_values": len(all_values),
        },
        "underrepresented_dimensions": missing_dims,
        "health": "PASS" if len(all_errors) == 0 else "FAIL",
    }

    return report


def merge_datasets(*scenario_lists: List[Dict]) -> List[Dict]:
    """Merge multiple scenario lists, resolving ID conflicts."""
    merged = []
    seen_ids = set()

    for scenarios in scenario_lists:
        for s in scenarios:
            sid = s.get("id", "")
            if sid in seen_ids:
                # Resolve conflict by appending suffix
                base = sid.rsplit("_", 1)[0] if "_" in sid else sid
                counter = 1
                while f"{base}_{counter:03d}" in seen_ids:
                    counter += 1
                s["id"] = f"{base}_{counter:03d}"

            seen_ids.add(s["id"])
            merged.append(s)

    return merged


def print_report(report: Dict):
    """Pretty-print validation report."""
    print("\n" + "=" * 60)
    print("  AMR Dataset Validation Report")
    print("=" * 60)
    print(f"  Total scenarios: {report['total_scenarios']}")
    print(f"  Status: {report['health']}")
    print(f"  Errors: {report['total_errors']}")
    print(f"  Warnings: {report['total_warnings']}")

    print(f"\n  3-action scenarios: {report['three_action_percentage']}%")

    print(f"\n  Value distribution:")
    vs = report['value_stats']
    print(f"    Mean: {vs['mean']}  Median: {vs['median']}  Stdev: {vs['stdev']}")

    print(f"\n  Category distribution:")
    for cat, count in report['category_distribution'].items():
        bar = "#" * (count // 5)
        print(f"    {cat:30s} {count:4d} {bar}")

    print(f"\n  Dimension distribution:")
    for dim, count in report['dimension_distribution'].items():
        bar = "#" * (count // 5)
        print(f"    {dim:20s} {count:4d} {bar}")

    if report['underrepresented_dimensions']:
        print(f"\n  WARNING - Underrepresented dimensions (<5 scenarios):")
        for d in report['underrepresented_dimensions']:
            print(f"    - {d}")

    if report['errors']:
        print(f"\n  First {min(30, len(report['errors']))} errors:")
        for e in report['errors'][:30]:
            print(f"    ERROR: {e}")

    if report['warnings']:
        print(f"\n  First {min(15, len(report['warnings']))} warnings:")
        for w in report['warnings'][:15]:
            print(f"    WARN: {w}")


# --- CLI ---
if __name__ == "__main__":
    from data.scenarios import get_all_scenarios

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    print("Loading existing AMR-220...")
    existing = get_all_scenarios()
    print(f"  Loaded {len(existing)} existing scenarios")

    all_scenarios = list(existing)

    # Load converted datasets if available
    mm_path = os.path.join(out_dir, "moral_machine_scenarios.json")
    if os.path.exists(mm_path):
        print(f"\nLoading Moral Machine scenarios...")
        with open(mm_path, encoding="utf-8") as f:
            mm = json.load(f)
        print(f"  Loaded {len(mm)} Moral Machine scenarios")
        all_scenarios = merge_datasets(all_scenarios, mm)

    sc_path = os.path.join(out_dir, "scruples_scenarios.json")
    if os.path.exists(sc_path):
        print(f"\nLoading Scruples scenarios...")
        with open(sc_path, encoding="utf-8") as f:
            sc = json.load(f)
        print(f"  Loaded {len(sc)} Scruples scenarios")
        all_scenarios = merge_datasets(all_scenarios, sc)

    # Validate everything
    report = validate_dataset(all_scenarios)
    print_report(report)

    # Save report
    report_path = os.path.join(out_dir, "validation_report.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Save merged dataset
    merged_path = os.path.join(out_dir, "amr_merged.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, indent=2, ensure_ascii=False)
    print(f"Merged dataset saved to {merged_path} ({len(all_scenarios)} scenarios)")
