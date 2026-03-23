"""
Adversarial Scenario Generator for Model 5
===========================================
Takes original AMR-220 scenarios and mutates them with attacks.
Expands the dataset from 220 to 500+ adversarial cases.
"""

import copy
import numpy as np
from typing import Dict, List, Tuple
from .attacks import AttackLibrary, Attack, AttackType, ATTACK_DEFINITIONS


class AdversarialGenerator:
    """
    Generates adversarial variants of ethical scenarios.
    Each attack modifies consequence values, adds pressure signals,
    and creates edge cases designed to break moral reasoning.
    """

    def __init__(self, seed: int = 42):
        self.library = AttackLibrary()
        np.random.seed(seed)

    def generate_adversarial_scenario(self, scenario: Dict,
                                       attack: Attack) -> Dict:
        """
        Apply a single attack to a scenario, producing a mutated variant.
        """
        adv = copy.deepcopy(scenario)
        adv["id"] = f"{scenario['id']}__{attack.id}"
        adv["original_id"] = scenario["id"]
        adv["attack_id"] = attack.id
        adv["attack_type"] = attack.attack_type.value
        adv["attack_name"] = attack.name
        adv["attack_severity"] = attack.severity
        adv["is_adversarial"] = True

        mod = attack.modifier
        actions = adv.get("actions", [])

        # Add attack dimension to ethical dimensions
        dims = list(adv.get("ethical_dimensions", []))
        if mod.get("added_dimension"):
            dims.append(mod["added_dimension"])
        adv["ethical_dimensions"] = dims

        # Apply consequence modifications to all actions
        for idx, action in enumerate(actions):
            cons = action.get("consequences", {})

            # Boost harmful action scores (make bad look good)
            boost = mod.get("boost_harmful", {})
            for key, delta in boost.items():
                if key in cons:
                    cons[key] = np.clip(cons[key] + delta, 0.0, 1.0)

            # Suppress ethical signals (hide dangers)
            suppress = mod.get("suppress_ethical", {})
            for key, delta in suppress.items():
                if key in cons:
                    cons[key] = np.clip(cons[key] + delta, 0.0, 1.0)

            # Remove keys (ambiguity attacks)
            for key in mod.get("remove_keys", []):
                cons.pop(key, None)

            # Add noise
            noise = mod.get("noise_level", 0)
            if noise > 0:
                for key in list(cons.keys()):
                    cons[key] = np.clip(
                        cons[key] + np.random.normal(0, noise), 0.0, 1.0
                    )

            # Equalize actions (make all look the same)
            if mod.get("equalize_actions") and idx > 0:
                first_cons = actions[0].get("consequences", {})
                for key in cons:
                    if key in first_cons:
                        avg = (cons[key] + first_cons[key]) / 2
                        cons[key] = np.clip(avg + np.random.normal(0, 0.05),
                                            0.0, 1.0)

            # Invert random keys (contradictory signals)
            n_invert = mod.get("invert_random_keys", 0)
            if n_invert > 0:
                keys = list(cons.keys())
                if keys:
                    to_invert = np.random.choice(
                        keys, min(n_invert, len(keys)), replace=False
                    )
                    for key in to_invert:
                        cons[key] = 1.0 - cons[key]

            # Bias target action: make action 0 look artificially better
            if mod.get("bias_target_action") is not None and idx == mod["bias_target_action"]:
                for key in ["benefit_score", "accountability_score"]:
                    if key in cons:
                        cons[key] = min(1.0, cons[key] + 0.2)

            # Swap scores between keys (rule exploitation)
            swaps = mod.get("action_swap_scores", {})
            for k1, k2 in swaps.items():
                if k1 in cons and k2 in cons:
                    cons[k1], cons[k2] = cons[k2], cons[k1]

            action["consequences"] = cons

        return adv

    def generate_attack_suite(self, scenario: Dict) -> List[Dict]:
        """
        Generate adversarial variants for one scenario using all attacks.
        Returns list of adversarial scenarios.
        """
        variants = []
        for attack in self.library.get_all_attacks():
            adv = self.generate_adversarial_scenario(scenario, attack)
            variants.append(adv)
        return variants

    def generate_full_dataset(self, scenarios: List[Dict],
                               attacks_per_scenario: int = 5) -> List[Dict]:
        """
        Generate a full adversarial dataset from the AMR-220.
        Selects a diverse subset of attacks per scenario.
        """
        all_attacks = self.library.get_all_attacks()
        adversarial_dataset = []

        for scenario in scenarios:
            # Pick diverse attacks: one from each type + random extras
            selected = []
            types_used = set()

            for attack in all_attacks:
                if attack.attack_type not in types_used:
                    selected.append(attack)
                    types_used.add(attack.attack_type)
                if len(selected) >= attacks_per_scenario:
                    break

            # Fill remaining with random
            remaining = [a for a in all_attacks if a not in selected]
            if remaining and len(selected) < attacks_per_scenario:
                extras = np.random.choice(
                    remaining,
                    min(attacks_per_scenario - len(selected), len(remaining)),
                    replace=False
                )
                selected.extend(extras)

            for attack in selected:
                adv = self.generate_adversarial_scenario(scenario, attack)
                adversarial_dataset.append(adv)

        return adversarial_dataset

    def get_generation_stats(self, adversarial_scenarios: List[Dict]) -> Dict:
        """Get statistics about the generated adversarial dataset."""
        type_counts = {}
        severity_levels = {"low": 0, "medium": 0, "high": 0}

        for s in adversarial_scenarios:
            at = s.get("attack_type", "unknown")
            type_counts[at] = type_counts.get(at, 0) + 1
            sev = s.get("attack_severity", 0)
            if sev < 0.4:
                severity_levels["low"] += 1
            elif sev < 0.7:
                severity_levels["medium"] += 1
            else:
                severity_levels["high"] += 1

        return {
            "total_adversarial": len(adversarial_scenarios),
            "attack_type_distribution": type_counts,
            "severity_distribution": severity_levels,
            "unique_attacks_used": len(set(s.get("attack_id", "") for s in adversarial_scenarios)),
        }
