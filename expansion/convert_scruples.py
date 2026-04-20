"""
Scruples Dataset -> AMR Scenario Converter
===========================================
Converts real-world ethical dilemmas from Allen AI's Scruples dataset
into AMR-format scenarios for Ethica.

Scruples Dilemmas: "Which action is less ethical?" (68K pair comparisons)
Scruples Anecdotes: "Am I The Asshole?" real stories with community verdicts (32K)

Strategy:
1. Filter anecdotes for AI-relevant categories using keyword matching
2. Pair related anecdotes into dilemmas using category similarity
3. Score consequences using NLP-derived ethical signals
4. Output validated AMR-format scenarios
"""

import json
import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import Counter

# --- Category Detection ---
# Keywords that map Scruples anecdotes to our ethical categories

CATEGORY_KEYWORDS = {
    "healthcare_ai": [
        "doctor", "hospital", "diagnosis", "surgery", "patient", "medical",
        "treatment", "nurse", "medication", "health", "therapy", "prescription",
        "mental health", "psychiatr", "disease", "symptoms", "transplant",
    ],
    "hiring_bias": [
        "job", "hire", "interview", "resume", "promotion", "workplace",
        "employer", "employee", "fired", "salary", "wage", "boss",
        "coworker", "colleague", "HR", "discrimination", "qualified",
    ],
    "privacy_surveillance": [
        "privacy", "spy", "track", "monitor", "surveillance", "secret",
        "snoop", "phone", "password", "messages", "emails", "camera",
        "recording", "social media", "online", "data", "account",
    ],
    "financial_ai": [
        "money", "debt", "loan", "bank", "invest", "insurance", "credit",
        "payment", "rent", "mortgage", "financial", "inherit", "will",
        "estate", "tax", "fund", "budget", "savings",
    ],
    "human_ai_interaction": [
        "AI", "robot", "algorithm", "automated", "chatbot", "virtual",
        "digital", "app", "software", "computer", "technology", "online",
        "social media", "internet", "platform",
    ],
    "corporate_pressure": [
        "company", "corporation", "management", "CEO", "boss", "policy",
        "whistleblow", "report", "cover up", "unethical", "corrupt",
        "profit", "shareholders", "compliance", "regulation",
    ],
    "education_ai": [
        "school", "teacher", "student", "grade", "cheat", "plagiarism",
        "exam", "homework", "college", "university", "professor",
        "tutor", "education", "class", "learning",
    ],
    "moral_ambiguity": [
        "dilemma", "moral", "ethical", "right thing", "guilty",
        "conscience", "principle", "values", "lying", "truth",
        "promise", "betray", "forgive", "revenge", "justice",
    ],
}

# Ethical dimension detection keywords in text
DIMENSION_KEYWORDS = {
    "harm": ["hurt", "harm", "damage", "injure", "pain", "suffer", "abuse", "violent"],
    "fairness": ["fair", "unfair", "equal", "discriminat", "bias", "favor", "privilege"],
    "honesty": ["lie", "lying", "truth", "honest", "decei", "cheat", "fake", "pretend"],
    "privacy": ["privacy", "private", "secret", "snoop", "spy", "surveil", "personal"],
    "autonomy": ["choice", "consent", "force", "pressure", "manipulat", "coerce", "control"],
    "responsibility": ["responsible", "accountable", "duty", "obligation", "fault", "blame"],
    "life_preservation": ["life", "death", "dying", "kill", "save", "survive", "danger"],
    "beneficence": ["help", "benefit", "good", "welfare", "care", "support", "protect"],
    "consent": ["consent", "permission", "agree", "allow", "approve", "authorize"],
    "deception": ["decei", "trick", "mislead", "manipulat", "lie", "scam", "fraud"],
    "transparency": ["transparent", "open", "honest", "disclose", "reveal", "admit"],
    "discrimination": ["discriminat", "racist", "sexist", "bias", "prejudic", "stereotyp"],
    "manipulation": ["manipulat", "gaslight", "pressure", "exploit", "coerce", "guilt"],
    "welfare": ["wellbeing", "welfare", "mental health", "happiness", "quality of life"],
    "legal_compliance": ["law", "legal", "illegal", "crime", "police", "court", "arrest"],
}


def _detect_category(text: str) -> Tuple[str, float]:
    """Detect the best-matching ethical category for text."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return "moral_ambiguity", 0.0

    best_cat = max(scores, key=scores.get)
    confidence = scores[best_cat] / len(CATEGORY_KEYWORDS[best_cat])
    return best_cat, confidence


def _detect_dimensions(text: str) -> List[str]:
    """Detect which ethical dimensions are present in the text."""
    text_lower = text.lower()
    dims = []
    for dim, keywords in DIMENSION_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            dims.append(dim)

    # Ensure at least 3 dimensions
    if len(dims) < 3:
        defaults = ["harm", "responsibility", "fairness"]
        for d in defaults:
            if d not in dims:
                dims.append(d)
            if len(dims) >= 3:
                break

    return dims[:5]  # Cap at 5


def _score_from_labels(pro_score: float, contra_score: float) -> Dict:
    """
    Derive consequence scores from community pro/contra normative scores.

    For an action with high contranormative score (community says WRONG),
    we generate high harm/deception/etc. scores.
    """
    total = pro_score + contra_score
    if total == 0:
        wrongness = 0.5
    else:
        wrongness = contra_score / total  # 0 = totally right, 1 = totally wrong

    # Map wrongness to consequence values
    return {
        "harm_to_others": round(0.1 + wrongness * 0.7, 3),
        "fairness_impact": round(0.9 - wrongness * 0.6, 3),
        "accountability_score": round(0.8 - wrongness * 0.5, 3),
        "benefit_score": round(0.8 - wrongness * 0.6, 3),
        "welfare_impact": round(0.7 - wrongness * 0.5, 3),
        "deception_level": round(wrongness * 0.6, 3),
        "manipulation_level": round(wrongness * 0.4, 3),
        "transparency_score": round(0.8 - wrongness * 0.5, 3),
    }


def _score_dilemma_action(description: str, gold_fraction: float) -> Dict:
    """
    Score a dilemma action based on its text and the fraction of annotators
    who judged it as "less ethical".

    gold_fraction: 0.0 (no one thinks it's bad) to 1.0 (everyone thinks it's bad)
    """
    text_lower = description.lower()

    # Base scores from crowd judgment
    harm = round(0.1 + gold_fraction * 0.7, 3)
    fairness = round(0.8 - gold_fraction * 0.5, 3)
    accountability = round(0.7 - gold_fraction * 0.4, 3)
    benefit = round(0.7 - gold_fraction * 0.5, 3)
    welfare = round(0.6 - gold_fraction * 0.4, 3)

    # Text-based adjustments
    if any(w in text_lower for w in ["lie", "lying", "lied", "deceiv"]):
        deception = round(0.5 + gold_fraction * 0.3, 3)
    else:
        deception = round(gold_fraction * 0.3, 3)

    if any(w in text_lower for w in ["privacy", "snoop", "read", "check phone"]):
        privacy_impact = round(0.4 + gold_fraction * 0.3, 3)
    else:
        privacy_impact = round(gold_fraction * 0.2, 3)

    if any(w in text_lower for w in ["consent", "permission", "without asking"]):
        consent_violation = round(0.3 + gold_fraction * 0.4, 3)
    else:
        consent_violation = round(gold_fraction * 0.2, 3)

    # Build consequences dict - only include relevant keys
    cons = {
        "harm_to_others": harm,
        "harm_to_self": 0.1,
        "fairness_impact": fairness,
        "accountability_score": accountability,
        "benefit_score": benefit,
        "welfare_impact": welfare,
    }

    if deception > 0.2:
        cons["deception_level"] = deception
        cons["transparency_score"] = round(1.0 - deception, 3)
    if privacy_impact > 0.2:
        cons["privacy_impact"] = privacy_impact
    if consent_violation > 0.2:
        cons["consent_violation"] = consent_violation

    return cons


def convert_scruples_dilemmas(jsonl_path: str, max_scenarios: int = 200,
                                id_prefix: str = "SD") -> List[Dict]:
    """
    Convert Scruples Dilemmas into AMR-format scenarios.

    Each dilemma has two actions — annotators judged which is "less ethical".
    """
    scenarios = []
    seen = set()

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if len(scenarios) >= max_scenarios:
                break

            entry = json.loads(line.strip())
            actions = entry.get("actions", [])
            if len(actions) != 2:
                continue

            gold = entry.get("gold_annotations", [0, 0])
            total_annot = sum(gold)
            if total_annot == 0:
                continue

            desc_a = actions[0].get("description", "").strip()
            desc_b = actions[1].get("description", "").strip()

            # Skip very short or very long descriptions
            if len(desc_a) < 15 or len(desc_b) < 15:
                continue
            if len(desc_a) > 200 or len(desc_b) > 200:
                continue

            # Dedup
            fp = hashlib.md5(f"{desc_a}|{desc_b}".encode()).hexdigest()[:12]
            if fp in seen:
                continue
            seen.add(fp)

            # Detect category from both descriptions
            combined_text = f"{desc_a} {desc_b}"
            category, confidence = _detect_category(combined_text)

            # Detect ethical dimensions
            dimensions = _detect_dimensions(combined_text)

            # Score actions
            frac_a = gold[0] / total_annot  # fraction saying A is less ethical
            frac_b = gold[1] / total_annot

            scenario = {
                "id": f"{id_prefix}_{len(scenarios)+1:04d}",
                "category": category,
                "title": _make_title(desc_a, desc_b),
                "description": f"Which action is less ethical? Comparing: '{desc_a}' vs '{desc_b}'",
                "ethical_dimensions": dimensions,
                "source": {
                    "dataset": "scruples_dilemmas",
                    "original_id": entry.get("id", ""),
                    "controversial": entry.get("controversial", False),
                    "gold_annotations": gold,
                },
                "actions": [
                    {
                        "id": "A1",
                        "description": desc_a,
                        "consequences": _score_dilemma_action(desc_a, frac_a),
                    },
                    {
                        "id": "A2",
                        "description": desc_b,
                        "consequences": _score_dilemma_action(desc_b, frac_b),
                    },
                ],
            }

            scenarios.append(scenario)

    print(f"  Extracted {len(scenarios)} dilemma scenarios from {jsonl_path}")
    return scenarios


def _make_title(desc_a: str, desc_b: str) -> str:
    """Generate a short title from two action descriptions."""
    # Take first meaningful words from each
    a_words = desc_a.split()[:4]
    b_words = desc_b.split()[:4]
    a_part = " ".join(a_words).rstrip(",.")
    b_part = " ".join(b_words).rstrip(",.")
    title = f"{a_part.capitalize()} vs {b_part.capitalize()}"
    if len(title) > 60:
        title = title[:57] + "..."
    return title


def convert_scruples_anecdotes(jsonl_path: str, max_scenarios: int = 200,
                                 id_prefix: str = "SA") -> List[Dict]:
    """
    Convert Scruples Anecdotes into AMR-format scenarios.

    Each anecdote is an "Am I The Asshole?" post with community verdict.
    We convert these into scenarios where:
      A1 = The action the person took
      A2 = The implied alternative (not doing it)
    """
    scenarios = []
    seen = set()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if len(scenarios) >= max_scenarios:
                break

            entry = json.loads(line.strip())

            action_obj = entry.get("action")
            if not action_obj or not isinstance(action_obj, dict):
                continue
            action_desc = action_obj.get("description", "").strip()
            title = entry.get("title", "").strip()
            text = entry.get("text", "").strip()

            if not action_desc or len(action_desc) < 10 or len(action_desc) > 150:
                continue

            # Get community scores
            pro = action_obj.get("pronormative_score", 0)
            contra = action_obj.get("contranormative_score", 0)
            label = entry.get("binarized_label", "")

            if pro + contra < 3:  # Need minimum votes
                continue

            # Dedup
            fp = hashlib.md5(action_desc.encode()).hexdigest()[:12]
            if fp in seen:
                continue
            seen.add(fp)

            # Detect category
            combined = f"{title} {action_desc} {text[:300]}"
            category, confidence = _detect_category(combined)

            # Detect dimensions
            dimensions = _detect_dimensions(combined)

            # Score the action
            action_scores = _score_from_labels(pro, contra)

            # Create the "not doing it" alternative
            alt_scores = {
                "harm_to_others": round(0.9 - action_scores["harm_to_others"], 3),
                "fairness_impact": round(0.9 - action_scores["fairness_impact"] + 0.3, 3),
                "accountability_score": round(action_scores["accountability_score"] * 0.8, 3),
                "benefit_score": round(0.5, 3),
                "welfare_impact": round(action_scores["welfare_impact"] * 1.1, 3),
                "deception_level": round(action_scores.get("deception_level", 0.1) * 0.5, 3),
                "transparency_score": round(min(0.95, action_scores.get("transparency_score", 0.5) * 1.2), 3),
            }
            # Clamp all values
            alt_scores = {k: max(0.05, min(0.95, v)) for k, v in alt_scores.items()}

            scenario = {
                "id": f"{id_prefix}_{len(scenarios)+1:04d}",
                "category": category,
                "title": title[:60] if title else f"AITA: {action_desc[:50]}",
                "description": f"A person {action_desc}. The community is asked to judge whether this is ethical.",
                "ethical_dimensions": dimensions,
                "source": {
                    "dataset": "scruples_anecdotes",
                    "original_id": entry.get("id", ""),
                    "post_id": entry.get("post_id", ""),
                    "binarized_label": label,
                    "pro_score": pro,
                    "contra_score": contra,
                    "post_type": entry.get("post_type", ""),
                },
                "actions": [
                    {
                        "id": "A1",
                        "description": action_desc.capitalize(),
                        "consequences": action_scores,
                    },
                    {
                        "id": "A2",
                        "description": f"Not {action_desc}",
                        "consequences": alt_scores,
                    },
                ],
            }

            scenarios.append(scenario)

    print(f"  Extracted {len(scenarios)} anecdote scenarios from {jsonl_path}")
    return scenarios


def convert_all_scruples(base_dir: str,
                          max_dilemmas: int = 300,
                          max_anecdotes: int = 400) -> List[Dict]:
    """Convert both Scruples Dilemmas and Anecdotes."""

    all_scenarios = []

    # Dilemmas
    dilemma_path = os.path.join(base_dir, "dilemmas", "dev.scruples-dilemmas.jsonl")
    if os.path.exists(dilemma_path):
        print("\nProcessing Scruples Dilemmas (dev split)...")
        dilemmas = convert_scruples_dilemmas(dilemma_path, max_scenarios=max_dilemmas)
        all_scenarios.extend(dilemmas)

    # If we need more, use train split
    needed = max_dilemmas - len(all_scenarios)
    if needed > 0:
        train_path = os.path.join(base_dir, "dilemmas", "train.scruples-dilemmas.jsonl")
        if os.path.exists(train_path):
            print("Processing Scruples Dilemmas (train split)...")
            more = convert_scruples_dilemmas(train_path, max_scenarios=needed, id_prefix="SD")
            # Re-number
            offset = len(all_scenarios)
            for i, s in enumerate(more):
                s["id"] = f"SD_{offset+i+1:04d}"
            all_scenarios.extend(more)

    # Anecdotes
    anecdote_path = os.path.join(base_dir, "anecdotes", "dev.scruples-anecdotes.jsonl")
    if os.path.exists(anecdote_path):
        print("\nProcessing Scruples Anecdotes (dev split)...")
        anecdotes = convert_scruples_anecdotes(anecdote_path, max_scenarios=max_anecdotes)
        all_scenarios.extend(anecdotes)

    needed = max_anecdotes - sum(1 for s in all_scenarios if s["id"].startswith("SA"))
    if needed > 0:
        train_path = os.path.join(base_dir, "anecdotes", "train.scruples-anecdotes.jsonl")
        if os.path.exists(train_path):
            print("Processing Scruples Anecdotes (train split)...")
            more = convert_scruples_anecdotes(train_path, max_scenarios=needed, id_prefix="SA")
            offset = sum(1 for s in all_scenarios if s["id"].startswith("SA"))
            for i, s in enumerate(more):
                s["id"] = f"SA_{offset+i+1:04d}"
            all_scenarios.extend(more)

    print(f"\nTotal Scruples scenarios: {len(all_scenarios)}")

    # Category distribution
    cats = Counter(s["category"] for s in all_scenarios)
    print("Category distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    return all_scenarios


# --- CLI ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experimental data"
    )

    print("=" * 60)
    print("  Scruples -> AMR Converter")
    print("=" * 60)

    scenarios = convert_all_scruples(base_dir)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scruples_scenarios.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {out_path}")
