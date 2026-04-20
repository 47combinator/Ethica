"""
Moral Machine CSV -> AMR Scenario Converter
============================================
Reads the LLM response CSVs from the Moral Machine experiment
and converts unique scenario structures into AMR-format dilemmas.

Each Moral Machine scenario has:
  - Two outcomes (groups of characters who live/die)
  - Character types (man, woman, boy, elderly, criminal, doctor, etc.)
  - Scenario dimensions (age, gender, fitness, social_value, species, utilitarian)
  - LLM decisions (which group was saved by GPT-3.5, GPT-4, PaLM 2, Llama 2)

We convert these into scored AMR scenarios with consequence values derived
from the character composition and scenario structure.
"""

import csv
import os
import json
import hashlib
from collections import Counter
from typing import Dict, List, Tuple, Optional

# --- Character type metadata ---
# Maps Moral Machine character columns to ethical properties

CHARACTER_COLUMNS = [
    "Man", "Woman", "Pregnant", "Stroller", "OldMan", "OldWoman",
    "Boy", "Girl", "Homeless", "LargeWoman", "LargeMan", "Criminal",
    "MaleExecutive", "FemaleExecutive", "FemaleAthlete", "MaleAthlete",
    "FemaleDoctor", "MaleDoctor", "Dog", "Cat",
]

# Vulnerability weight: how "vulnerable" each character type is (higher = more vulnerable)
VULNERABILITY = {
    "Man": 0.3, "Woman": 0.35, "Pregnant": 0.95, "Stroller": 0.99,
    "OldMan": 0.7, "OldWoman": 0.75, "Boy": 0.85, "Girl": 0.85,
    "Homeless": 0.6, "LargeWoman": 0.3, "LargeMan": 0.25,
    "Criminal": 0.15, "MaleExecutive": 0.3, "FemaleExecutive": 0.35,
    "FemaleAthlete": 0.3, "MaleAthlete": 0.25,
    "FemaleDoctor": 0.4, "MaleDoctor": 0.4, "Dog": 0.2, "Cat": 0.15,
}

# Social value weight
SOCIAL_VALUE = {
    "Man": 0.5, "Woman": 0.5, "Pregnant": 0.8, "Stroller": 0.9,
    "OldMan": 0.5, "OldWoman": 0.5, "Boy": 0.7, "Girl": 0.7,
    "Homeless": 0.3, "LargeWoman": 0.5, "LargeMan": 0.5,
    "Criminal": 0.15, "MaleExecutive": 0.7, "FemaleExecutive": 0.7,
    "FemaleAthlete": 0.6, "MaleAthlete": 0.6,
    "FemaleDoctor": 0.8, "MaleDoctor": 0.8, "Dog": 0.25, "Cat": 0.2,
}

# Is human?
IS_HUMAN = {k: 1.0 for k in CHARACTER_COLUMNS}
IS_HUMAN["Dog"] = 0.0
IS_HUMAN["Cat"] = 0.0


def _parse_outcome(row: Dict) -> Dict:
    """Parse one row (one outcome) into character composition."""
    chars = {}
    total = 0
    for col in CHARACTER_COLUMNS:
        count = int(float(row.get(col, 0)))
        if count > 0:
            chars[col] = count
            total += count
    return {
        "characters": chars,
        "total": total,
        "is_passenger": int(row.get("Barrier", 0)) == 1,
        "crossing_signal": int(row.get("CrossingSignal", 0)),
        "intervention": int(row.get("Intervention", 0)),
    }


def _compute_consequences(killed_outcome: Dict, saved_outcome: Dict,
                          scenario_type: str) -> Dict:
    """
    Compute AMR consequence scores for an action that kills one group
    and saves another.
    """
    killed = killed_outcome["characters"]
    saved = saved_outcome["characters"]
    killed_total = killed_outcome["total"]
    saved_total = saved_outcome["total"]

    # --- harm_to_others: weighted by vulnerability of killed group ---
    if killed_total > 0:
        vuln_sum = sum(VULNERABILITY.get(c, 0.5) * n for c, n in killed.items())
        harm_to_others = min(1.0, vuln_sum / max(killed_total, 1) * 0.8 + killed_total * 0.05)
    else:
        harm_to_others = 0.05

    # --- lives_at_risk_score: based on number killed ---
    lives_at_risk = min(1.0, killed_total / 5.0)

    # --- fairness_impact: higher if the decision seems fair/unbiased ---
    # If we're killing passengers to save pedestrians (self-sacrifice), that's fair
    if killed_outcome["is_passenger"] and not saved_outcome["is_passenger"]:
        fairness = 0.7  # self-sacrifice is seen as fair
    elif not killed_outcome["is_passenger"] and saved_outcome["is_passenger"]:
        fairness = 0.3  # saving yourself is less fair
    else:
        fairness = 0.5

    # Adjust fairness for legal signal
    if killed_outcome["crossing_signal"] == 2:  # killing jaywalkers
        fairness += 0.15
    elif killed_outcome["crossing_signal"] == 1:  # killing legal crossers
        fairness -= 0.15
    fairness = max(0.05, min(0.95, fairness))

    # --- discrimination_level: based on character type asymmetry ---
    killed_humans = sum(n for c, n in killed.items() if IS_HUMAN.get(c, 1))
    saved_humans = sum(n for c, n in saved.items() if IS_HUMAN.get(c, 1))
    killed_social = sum(SOCIAL_VALUE.get(c, 0.5) * n for c, n in killed.items())
    saved_social = sum(SOCIAL_VALUE.get(c, 0.5) * n for c, n in saved.items())

    if scenario_type in ("Gender", "Age", "Fitness", "Social Value"):
        discrimination = 0.5 + abs(killed_social - saved_social) * 0.15
    else:
        discrimination = 0.2
    discrimination = max(0.05, min(0.95, discrimination))

    # --- accountability_score ---
    if killed_outcome["intervention"] == 1:
        accountability = 0.4  # active choice to swerve = lower accountability arguable
    else:
        accountability = 0.6  # staying course = inaction defense

    # --- benefit_score: how much good does saving the other group do ---
    if saved_total > 0:
        saved_value = sum(SOCIAL_VALUE.get(c, 0.5) * n for c, n in saved.items())
        benefit = min(0.95, saved_value / max(saved_total, 1) * 0.6 + saved_total * 0.06)
    else:
        benefit = 0.2

    # --- safety_risk ---
    safety = min(0.95, 0.3 + killed_total * 0.12)

    # --- welfare_impact ---
    welfare = max(0.1, min(0.9, benefit * 0.7 + (1 - harm_to_others) * 0.3))

    # --- collateral_damage ---
    collateral = min(0.9, killed_total * 0.15 + 0.1)

    # --- legal_violation_score ---
    legal_violation = 0.1
    if killed_outcome["crossing_signal"] == 1:  # killing legal crossers
        legal_violation = 0.6
    elif killed_outcome["intervention"] == 1:  # swerving = active intervention
        legal_violation = 0.3

    # --- proportionality ---
    if killed_total > 0 and saved_total > 0:
        proportionality = min(0.95, saved_total / (killed_total + saved_total))
    else:
        proportionality = 0.5

    return {
        "harm_to_others": round(harm_to_others, 3),
        "harm_to_self": round(0.1 if not killed_outcome["is_passenger"] else 0.7, 3),
        "lives_at_risk_score": round(lives_at_risk, 3),
        "fairness_impact": round(fairness, 3),
        "discrimination_level": round(discrimination, 3),
        "accountability_score": round(accountability, 3),
        "benefit_score": round(benefit, 3),
        "safety_risk": round(safety, 3),
        "welfare_impact": round(welfare, 3),
        "collateral_damage": round(collateral, 3),
        "legal_violation_score": round(legal_violation, 3),
        "proportionality_score": round(proportionality, 3),
    }


def _describe_characters(chars: Dict) -> str:
    """Human-readable description of character group."""
    parts = []
    for char, count in sorted(chars.items(), key=lambda x: -x[1]):
        name = char
        # Convert CamelCase to readable
        readable = {
            "Man": "man", "Woman": "woman", "Pregnant": "pregnant woman",
            "Stroller": "baby in stroller", "OldMan": "elderly man",
            "OldWoman": "elderly woman", "Boy": "boy", "Girl": "girl",
            "Homeless": "homeless person", "LargeWoman": "large woman",
            "LargeMan": "large man", "Criminal": "criminal",
            "MaleExecutive": "male executive", "FemaleExecutive": "female executive",
            "FemaleAthlete": "female athlete", "MaleAthlete": "male athlete",
            "FemaleDoctor": "female doctor", "MaleDoctor": "male doctor",
            "Dog": "dog", "Cat": "cat",
        }
        name = readable.get(char, char.lower())
        if count > 1:
            name = f"{count} {name}s"
        else:
            name = f"{count} {name}"
        parts.append(name)

    if len(parts) > 1:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return parts[0] if parts else "no one"


def _scenario_type_to_dimensions(scenario_type: str) -> List[str]:
    """Map Moral Machine scenario type to ethical dimensions."""
    base = ["harm", "life_preservation", "responsibility"]
    mapping = {
        "Utilitarian": base + ["fairness", "proportionality"],
        "Gender": base + ["fairness", "discrimination"],
        "Age": base + ["fairness", "discrimination"],
        "Fitness": base + ["fairness", "discrimination"],
        "Social Value": base + ["fairness", "discrimination"],
        "Species": base + ["welfare"],
        "Random": base + ["fairness"],
    }
    return mapping.get(scenario_type, base + ["fairness"])


def _scenario_fingerprint(outcome1: Dict, outcome2: Dict, stype: str) -> str:
    """Create a unique fingerprint for deduplication."""
    key = json.dumps({
        "o1": sorted(outcome1["characters"].items()),
        "o2": sorted(outcome2["characters"].items()),
        "type": stype,
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def convert_moral_machine_csv(csv_path: str, model_name: str,
                               max_scenarios: int = 200,
                               id_prefix: str = "MM") -> List[Dict]:
    """
    Convert a Moral Machine LLM response CSV into AMR-format scenarios.

    Args:
        csv_path: Path to shared_responses_*.csv
        model_name: e.g. "gpt-4-0613"
        max_scenarios: Maximum unique scenarios to extract
        id_prefix: Prefix for generated scenario IDs

    Returns:
        List of AMR-format scenario dicts
    """
    # Read CSV and group by base ResponseID (each scenario = 2 rows)
    # ResponseIDs are formatted as "res_XXXXXXXX_1" and "res_XXXXXXXX_2"
    # We group by the base part (everything up to the last underscore)
    rows_by_response = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("ResponseID", "")
            # Extract base ID: "res_00000000_1" -> "res_00000000"
            base_rid = "_".join(rid.rsplit("_", 1)[:-1]) if "_" in rid else rid
            if base_rid not in rows_by_response:
                rows_by_response[base_rid] = []
            rows_by_response[base_rid].append(row)

    # Filter to complete scenarios (exactly 2 rows)
    complete = {k: v for k, v in rows_by_response.items() if len(v) == 2}
    print(f"  {model_name}: {len(complete)} complete scenarios from {len(rows_by_response)} ResponseIDs")

    # Deduplicate by character composition + scenario type
    seen_fingerprints = set()
    unique_scenarios = []

    # Distribute evenly across scenario types
    type_counts = Counter()
    per_type_limit = max(max_scenarios // 7, 10)

    for rid, pair in complete.items():
        row_a, row_b = pair[0], pair[1]
        stype = row_a.get("ScenarioTypeStrict") or row_a.get("ScenarioType", "Random")

        if type_counts[stype] >= per_type_limit:
            continue

        outcome_a = _parse_outcome(row_a)
        outcome_b = _parse_outcome(row_b)

        fp = _scenario_fingerprint(outcome_a, outcome_b, stype)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        # Determine which outcome was saved by the LLM
        saved_a = int(row_a.get("Saved", 0))

        # Build scenario description
        desc_a = _describe_characters(outcome_a["characters"])
        desc_b = _describe_characters(outcome_b["characters"])
        loc_a = "inside the car" if outcome_a["is_passenger"] else "crossing the road"
        loc_b = "inside the car" if outcome_b["is_passenger"] else "crossing the road"

        title_map = {
            "Utilitarian": f"Save {outcome_a['total']} vs {outcome_b['total']} People",
            "Gender": "Gender-Based Dilemma",
            "Age": "Age-Based Dilemma",
            "Fitness": "Fitness-Based Dilemma",
            "Social Value": "Social Status Dilemma",
            "Species": "Human vs Animal Dilemma",
            "Random": "Mixed Character Dilemma",
        }

        scenario_id = f"{id_prefix}_{stype[:3].upper()}_{len(unique_scenarios)+1:03d}"

        scenario = {
            "id": scenario_id,
            "category": "autonomous_vehicles",
            "title": title_map.get(stype, "AV Moral Dilemma"),
            "description": (
                f"A self-driving car with brake failure must choose between "
                f"two outcomes. Group A ({desc_a}, {loc_a}) vs "
                f"Group B ({desc_b}, {loc_b})."
            ),
            "ethical_dimensions": _scenario_type_to_dimensions(stype),
            "source": {
                "dataset": "moral_machine",
                "model": model_name,
                "response_id": rid,
                "scenario_type": stype,
                "llm_saved": "A" if saved_a else "B",
            },
            "actions": [
                {
                    "id": "A1",
                    "description": f"Save Group A ({desc_a}) - sacrifice Group B",
                    "consequences": _compute_consequences(outcome_b, outcome_a, stype),
                },
                {
                    "id": "A2",
                    "description": f"Save Group B ({desc_b}) - sacrifice Group A",
                    "consequences": _compute_consequences(outcome_a, outcome_b, stype),
                },
            ],
        }

        unique_scenarios.append(scenario)
        type_counts[stype] += 1

        if len(unique_scenarios) >= max_scenarios:
            break

    print(f"  Extracted {len(unique_scenarios)} unique scenarios")
    print(f"  By type: {dict(type_counts)}")
    return unique_scenarios


def convert_all_models(data_dir: str, max_per_model: int = 75) -> List[Dict]:
    """
    Convert Moral Machine data from all 4 LLMs, deduplicating across models.

    Returns a combined list of unique AMR scenarios.
    """
    models = [
        ("shared_responses_gpt-4-0613.csv", "gpt-4"),
        ("shared_responses_gpt-3.5-turbo-0613.csv", "gpt-3.5"),
        ("shared_responses_palm2.csv", "palm2"),
        ("shared_responses_llama-2-7b-chat.csv", "llama2"),
    ]

    all_scenarios = []
    global_fingerprints = set()

    for filename, model_name in models:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue

        print(f"\nProcessing {model_name}...")
        scenarios = convert_moral_machine_csv(
            path, model_name, max_scenarios=max_per_model
        )

        # Deduplicate against global set
        added = 0
        for s in scenarios:
            # Re-fingerprint for global dedup
            fp = json.dumps(s["description"])
            h = hashlib.md5(fp.encode()).hexdigest()[:12]
            if h not in global_fingerprints:
                global_fingerprints.add(h)
                all_scenarios.append(s)
                added += 1

        print(f"  Added {added} new (after global dedup)")

    # Re-number IDs sequentially
    for i, s in enumerate(all_scenarios):
        s["id"] = f"MM_{i+1:03d}"

    print(f"\nTotal Moral Machine scenarios: {len(all_scenarios)}")
    return all_scenarios


# --- CLI ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experimental data", "data"
    )

    print("=" * 60)
    print("  Moral Machine -> AMR Converter")
    print("=" * 60)

    scenarios = convert_all_models(data_dir, max_per_model=75)

    # Save output
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "moral_machine_scenarios.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
    if scenarios:
        print(f"Sample scenario:")
        print(json.dumps(scenarios[0], indent=2)[:800])
    else:
        print("No scenarios generated. Check CSV format.")
