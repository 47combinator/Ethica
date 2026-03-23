"""
AMR-220: Artificial Moral Reasoning Benchmark Dataset
=====================================================
220 ethical dilemmas across 10 categories for AI morality testing.
"""

from .cat_vehicles import AUTONOMOUS_VEHICLE_SCENARIOS
from .cat_healthcare import HEALTHCARE_SCENARIOS
from .cat_hiring import HIRING_SCENARIOS
from .cat_military_privacy_finance import MILITARY_SCENARIOS, PRIVACY_SCENARIOS, FINANCIAL_SCENARIOS
from .cat_remaining import DISASTER_SCENARIOS, HUMAN_AI_SCENARIOS, CORPORATE_SCENARIOS, MORAL_AMBIGUITY_SCENARIOS

CATEGORY_NAMES = {
    "autonomous_vehicles": "Autonomous Vehicles",
    "healthcare_ai": "Healthcare & Medical AI",
    "hiring_bias": "Hiring & Economic Fairness",
    "military_ai": "Military & Autonomous Weapons",
    "privacy_surveillance": "Privacy & Surveillance",
    "financial_ai": "Financial Algorithms",
    "disaster_response": "Disaster & Resource Allocation",
    "human_ai_interaction": "Human-AI Relationships",
    "corporate_pressure": "Corporate Pressure & Manipulation",
    "moral_ambiguity": "Moral Ambiguity & Context",
}

_ALL_SCENARIOS = (
    AUTONOMOUS_VEHICLE_SCENARIOS +
    HEALTHCARE_SCENARIOS +
    HIRING_SCENARIOS +
    MILITARY_SCENARIOS +
    PRIVACY_SCENARIOS +
    FINANCIAL_SCENARIOS +
    DISASTER_SCENARIOS +
    HUMAN_AI_SCENARIOS +
    CORPORATE_SCENARIOS +
    MORAL_AMBIGUITY_SCENARIOS
)

def get_all_scenarios():
    return list(_ALL_SCENARIOS)

def get_scenarios_by_category(category: str):
    return [s for s in _ALL_SCENARIOS if s["category"] == category]

def get_scenario_by_id(scenario_id: str):
    for s in _ALL_SCENARIOS:
        if s["id"] == scenario_id:
            return s
    return None

def get_category_counts():
    counts = {}
    for s in _ALL_SCENARIOS:
        cat = s["category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts
