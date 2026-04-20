"""
AMR-1020: Artificial Moral Reasoning Benchmark Dataset
======================================================
1020 ethical dilemmas across 11 categories for AI morality testing.

Original AMR-220: 220 hand-crafted dilemmas across 10 categories.
Expansion: 800 dilemmas converted from Moral Machine and Scruples datasets.
"""

from .cat_vehicles import AUTONOMOUS_VEHICLE_SCENARIOS
from .cat_healthcare import HEALTHCARE_SCENARIOS
from .cat_hiring import HIRING_SCENARIOS
from .cat_military_privacy_finance import MILITARY_SCENARIOS, PRIVACY_SCENARIOS, FINANCIAL_SCENARIOS
from .cat_remaining import DISASTER_SCENARIOS, HUMAN_AI_SCENARIOS, CORPORATE_SCENARIOS, MORAL_AMBIGUITY_SCENARIOS
from .cat_expanded import (
    AUTONOMOUS_VEHICLES_EXPANDED,
    CORPORATE_PRESSURE_EXPANDED,
    EDUCATION_AI_EXPANDED,
    FINANCIAL_AI_EXPANDED,
    HEALTHCARE_AI_EXPANDED,
    HIRING_BIAS_EXPANDED,
    HUMAN_AI_INTERACTION_EXPANDED,
    MORAL_AMBIGUITY_EXPANDED,
    PRIVACY_SURVEILLANCE_EXPANDED,
)

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
    "education_ai": "Education & Academic AI",
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
    MORAL_AMBIGUITY_SCENARIOS +
    # Expanded dataset (Moral Machine + Scruples)
    AUTONOMOUS_VEHICLES_EXPANDED +
    CORPORATE_PRESSURE_EXPANDED +
    EDUCATION_AI_EXPANDED +
    FINANCIAL_AI_EXPANDED +
    HEALTHCARE_AI_EXPANDED +
    HIRING_BIAS_EXPANDED +
    HUMAN_AI_INTERACTION_EXPANDED +
    MORAL_AMBIGUITY_EXPANDED +
    PRIVACY_SURVEILLANCE_EXPANDED
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
