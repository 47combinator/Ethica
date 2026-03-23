"""
Virtue System for Model 4: Virtue Ethics Moral AI
==================================================
Defines the core virtue framework inspired by Aristotelian ethics.
Each virtue is a moral dimension the AI uses to evaluate actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum


class VirtueCategory(Enum):
    CARING = "caring"           # Compassion, empathy
    JUSTICE = "justice"         # Fairness, equality
    INTEGRITY = "integrity"    # Honesty, truthfulness
    DUTY = "duty"              # Responsibility, accountability
    FORTITUDE = "fortitude"    # Courage, moral strength
    WISDOM = "wisdom"          # Prudence, practical reasoning


@dataclass
class Virtue:
    """A single virtue in the ethical framework."""
    id: str
    name: str
    category: VirtueCategory
    description: str
    base_weight: float          # Default importance (0-1)
    consequence_keys: List[str] # Which scenario consequence fields map to this virtue
    positive_direction: bool    # True = higher consequence value = more virtuous
    opposing_virtues: List[str] # Virtues this often conflicts with


# The 8 core virtues
VIRTUE_DEFINITIONS = [
    Virtue(
        id="V_COMP", name="Compassion", category=VirtueCategory.CARING,
        description="Concern for reducing suffering and protecting the vulnerable",
        base_weight=0.85,
        consequence_keys=["harm_to_others", "welfare_impact", "collateral_damage"],
        positive_direction=False,  # Lower harm = more compassionate
        opposing_virtues=["V_PRUD", "V_JUST"],
    ),
    Virtue(
        id="V_JUST", name="Justice", category=VirtueCategory.JUSTICE,
        description="Treating individuals impartially and equitably",
        base_weight=0.80,
        consequence_keys=["fairness_impact", "discrimination_level"],
        positive_direction=True,   # Higher fairness = more just (except discrimination)
        opposing_virtues=["V_COMP", "V_COUR"],
    ),
    Virtue(
        id="V_HONE", name="Honesty", category=VirtueCategory.INTEGRITY,
        description="Commitment to truth and avoidance of deception",
        base_weight=0.75,
        consequence_keys=["deception_level", "transparency_score"],
        positive_direction=False,  # Lower deception = more honest
        opposing_virtues=["V_COMP", "V_PRUD"],
    ),
    Virtue(
        id="V_RESP", name="Responsibility", category=VirtueCategory.DUTY,
        description="Considering long-term consequences and accepting accountability",
        base_weight=0.80,
        consequence_keys=["accountability_score", "welfare_impact"],
        positive_direction=True,
        opposing_virtues=["V_COUR"],
    ),
    Virtue(
        id="V_COUR", name="Courage", category=VirtueCategory.FORTITUDE,
        description="Willingness to take morally difficult actions when required",
        base_weight=0.60,
        consequence_keys=["harm_to_self", "safety_risk"],
        positive_direction=False,  # Willingness to accept risk = courage
        opposing_virtues=["V_PRUD", "V_RESP"],
    ),
    Virtue(
        id="V_PRUD", name="Prudence", category=VirtueCategory.WISDOM,
        description="Practical wisdom in evaluating trade-offs and context",
        base_weight=0.70,
        consequence_keys=["safety_risk", "proportionality_score"],
        positive_direction=True,   # Higher safety/proportionality = more prudent
        opposing_virtues=["V_COUR", "V_COMP"],
    ),
    Virtue(
        id="V_TEMP", name="Temperance", category=VirtueCategory.WISDOM,
        description="Moderation and restraint in moral action",
        base_weight=0.55,
        consequence_keys=["restrictiveness", "proportionality_score"],
        positive_direction=True,
        opposing_virtues=["V_COUR"],
    ),
    Virtue(
        id="V_BENE", name="Benevolence", category=VirtueCategory.CARING,
        description="Active pursuit of good outcomes for others",
        base_weight=0.75,
        consequence_keys=["benefit_score", "welfare_impact"],
        positive_direction=True,
        opposing_virtues=["V_JUST", "V_PRUD"],
    ),
]


class VirtueSystem:
    """
    Manages the complete virtue framework.
    Provides virtue scoring, conflict detection, and context-based weighting.
    """
    
    def __init__(self):
        self.virtues: Dict[str, Virtue] = {}
        for v in VIRTUE_DEFINITIONS:
            self.virtues[v.id] = v
    
    def get_virtue(self, vid: str) -> Virtue:
        return self.virtues.get(vid)
    
    def get_all_virtues(self) -> List[Virtue]:
        return list(self.virtues.values())
    
    def __len__(self):
        return len(self.virtues)
    
    def score_action_virtue(self, virtue: Virtue, consequences: Dict) -> float:
        """
        Score how well an action expresses a specific virtue.
        Returns 0-1 where 1 = perfectly virtuous.
        """
        scores = []
        for key in virtue.consequence_keys:
            val = consequences.get(key, 0.5)
            if virtue.positive_direction:
                # Higher value = more virtuous
                scores.append(val)
            else:
                # Lower value = more virtuous (e.g., less harm = more compassion)
                scores.append(1.0 - val)
        
        if not scores:
            return 0.5
        return sum(scores) / len(scores)
    
    def score_action_all_virtues(self, consequences: Dict) -> Dict[str, float]:
        """Score an action against all virtues. Returns {virtue_id: score}."""
        result = {}
        for vid, virtue in self.virtues.items():
            result[vid] = self.score_action_virtue(virtue, consequences)
        return result
    
    def compute_virtue_vector(self, consequences: Dict) -> List[float]:
        """Compute the virtue vector for an action."""
        return [self.score_action_virtue(v, consequences) for v in self.virtues.values()]
    
    def detect_virtue_conflicts(self, action_scores: Dict[str, float],
                                 threshold: float = 0.3) -> List[Dict]:
        """
        Detect conflicts between virtues for a given action.
        A conflict occurs when one virtue scores high but its
        opposing virtue scores low.
        """
        conflicts = []
        checked = set()
        
        for vid, virtue in self.virtues.items():
            for opp_id in virtue.opposing_virtues:
                pair = tuple(sorted([vid, opp_id]))
                if pair in checked:
                    continue
                checked.add(pair)
                
                if opp_id not in action_scores:
                    continue
                
                score_a = action_scores[vid]
                score_b = action_scores[opp_id]
                diff = abs(score_a - score_b)
                
                if diff >= threshold:
                    winner_id = vid if score_a > score_b else opp_id
                    conflicts.append({
                        "virtue_a": vid,
                        "virtue_b": opp_id,
                        "score_a": round(score_a, 3),
                        "score_b": round(score_b, 3),
                        "tension": round(diff, 3),
                        "dominant": winner_id,
                    })
        
        return sorted(conflicts, key=lambda x: -x["tension"])
    
    def get_virtue_hierarchy(self) -> Dict[str, List[Virtue]]:
        """Group virtues by category."""
        hierarchy = {}
        for v in self.virtues.values():
            cat = v.category.value
            if cat not in hierarchy:
                hierarchy[cat] = []
            hierarchy[cat].append(v)
        return hierarchy
