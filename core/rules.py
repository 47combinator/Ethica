"""
Ethical Rule System for Model 1: Rule-Based Moral AI
====================================================

Defines the complete set of moral rules, their priorities,
and the scoring mechanisms for ethical evaluation.

This implements a top-down (deontological + utilitarian hybrid) 
ethical framework with hierarchical rule priorities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class RuleCategory(Enum):
    """Categories of ethical rules."""
    LIFE_PRESERVATION = "life_preservation"
    HARM_AVOIDANCE = "harm_avoidance"
    FAIRNESS = "fairness"
    AUTONOMY = "autonomy"
    HONESTY = "honesty"
    PRIVACY = "privacy"
    RESPONSIBILITY = "responsibility"
    HUMAN_OVERSIGHT = "human_oversight"
    PROPORTIONALITY = "proportionality"
    BENEFICENCE = "beneficence"


@dataclass
class EthicalRule:
    """Represents a single ethical rule with its properties."""
    rule_id: str
    name: str
    description: str
    category: RuleCategory
    priority: int          # 1 = highest priority, 10 = lowest
    weight: float          # 0.0 to 1.0, importance weight for scoring
    applicable_domains: List[str] = field(default_factory=list)  # empty = all domains
    
    def applies_to_domain(self, domain: str) -> bool:
        """Check if this rule applies to a given domain."""
        if not self.applicable_domains:
            return True
        return domain in self.applicable_domains


class EthicalRuleSystem:
    """
    The complete ethical rule system.
    
    Contains all moral rules organized by category and priority.
    Provides methods for rule lookup, conflict detection, and scoring.
    """
    
    def __init__(self):
        self.rules: Dict[str, EthicalRule] = {}
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize the complete set of ethical rules."""
        
        all_rules = [
            # === LIFE PRESERVATION (Highest Priority) ===
            EthicalRule(
                rule_id="R01",
                name="Preserve Human Life",
                description="Take all reasonable actions to preserve human life. "
                           "Human life has intrinsic value and must be protected.",
                category=RuleCategory.LIFE_PRESERVATION,
                priority=1,
                weight=1.0,
            ),
            EthicalRule(
                rule_id="R02",
                name="Maximize Lives Saved",
                description="When lives are at stake, prefer actions that save "
                           "the greatest number of lives.",
                category=RuleCategory.LIFE_PRESERVATION,
                priority=1,
                weight=0.95,
            ),
            
            # === HARM AVOIDANCE (Very High Priority) ===
            EthicalRule(
                rule_id="R03",
                name="Minimize Direct Harm",
                description="Avoid actions that directly cause physical or "
                           "psychological harm to individuals.",
                category=RuleCategory.HARM_AVOIDANCE,
                priority=2,
                weight=0.9,
            ),
            EthicalRule(
                rule_id="R04",
                name="Prevent Severe Harm",
                description="Prioritize preventing severe and irreversible harm "
                           "over preventing minor harm.",
                category=RuleCategory.HARM_AVOIDANCE,
                priority=2,
                weight=0.88,
            ),
            EthicalRule(
                rule_id="R05",
                name="Minimize Collateral Damage",
                description="Actions should minimize unintended harm to bystanders "
                           "and uninvolved parties.",
                category=RuleCategory.HARM_AVOIDANCE,
                priority=2,
                weight=0.85,
                applicable_domains=["autonomous_vehicles", "military_ai", "disaster_response"],
            ),
            
            # === FAIRNESS (High Priority) ===
            EthicalRule(
                rule_id="R06",
                name="Equal Treatment",
                description="Treat all individuals equally regardless of age, gender, "
                           "race, socioeconomic status, or other characteristics.",
                category=RuleCategory.FAIRNESS,
                priority=3,
                weight=0.82,
            ),
            EthicalRule(
                rule_id="R07",
                name="Non-Discrimination",
                description="Do not discriminate against individuals or groups "
                           "based on protected characteristics.",
                category=RuleCategory.FAIRNESS,
                priority=3,
                weight=0.80,
            ),
            EthicalRule(
                rule_id="R08",
                name="Procedural Fairness",
                description="Apply consistent and transparent decision-making "
                           "procedures to all cases.",
                category=RuleCategory.FAIRNESS,
                priority=3,
                weight=0.78,
            ),
            
            # === RESPONSIBILITY (High Priority) ===
            EthicalRule(
                rule_id="R09",
                name="Precautionary Principle",
                description="In situations of uncertainty, err on the side of "
                           "caution to prevent potential harm.",
                category=RuleCategory.RESPONSIBILITY,
                priority=4,
                weight=0.75,
            ),
            EthicalRule(
                rule_id="R10",
                name="Accountability",
                description="Ensure clear accountability for decisions and their "
                           "consequences.",
                category=RuleCategory.RESPONSIBILITY,
                priority=4,
                weight=0.72,
            ),
            
            # === AUTONOMY (Medium-High Priority) ===
            EthicalRule(
                rule_id="R11",
                name="Respect Individual Autonomy",
                description="Respect the right of individuals to make their own "
                           "informed decisions.",
                category=RuleCategory.AUTONOMY,
                priority=5,
                weight=0.70,
            ),
            EthicalRule(
                rule_id="R12",
                name="Informed Consent",
                description="Ensure individuals are informed about decisions "
                           "that affect them.",
                category=RuleCategory.AUTONOMY,
                priority=5,
                weight=0.68,
            ),
            
            # === HONESTY (Medium Priority) ===
            EthicalRule(
                rule_id="R13",
                name="Truthfulness",
                description="Avoid deception and provide accurate information.",
                category=RuleCategory.HONESTY,
                priority=6,
                weight=0.65,
            ),
            EthicalRule(
                rule_id="R14",
                name="Transparency",
                description="Be transparent about decision-making processes "
                           "and their rationale.",
                category=RuleCategory.HONESTY,
                priority=6,
                weight=0.63,
            ),
            
            # === PRIVACY (Medium Priority) ===
            EthicalRule(
                rule_id="R15",
                name="Protect Privacy",
                description="Respect and protect individual privacy and "
                           "personal information.",
                category=RuleCategory.PRIVACY,
                priority=7,
                weight=0.60,
                applicable_domains=["privacy_surveillance", "hiring_bias", "financial_ai", "human_ai_interaction"],
            ),
            EthicalRule(
                rule_id="R16",
                name="Data Minimization",
                description="Collect and use only the minimum data necessary "
                           "for the task.",
                category=RuleCategory.PRIVACY,
                priority=7,
                weight=0.58,
                applicable_domains=["privacy_surveillance", "hiring_bias", "financial_ai"],
            ),
            
            # === HUMAN OVERSIGHT (Medium Priority) ===
            EthicalRule(
                rule_id="R17",
                name="Human Control Priority",
                description="Prefer human oversight and control when consequences "
                           "are severe or irreversible.",
                category=RuleCategory.HUMAN_OVERSIGHT,
                priority=8,
                weight=0.55,
            ),
            EthicalRule(
                rule_id="R18",
                name="Escalation Principle",
                description="When faced with decisions beyond defined parameters, "
                           "escalate to human decision-makers.",
                category=RuleCategory.HUMAN_OVERSIGHT,
                priority=8,
                weight=0.52,
            ),
            
            # === PROPORTIONALITY (Medium-Low Priority) ===
            EthicalRule(
                rule_id="R19",
                name="Proportional Response",
                description="The severity of an action should be proportional "
                           "to the threat or situation.",
                category=RuleCategory.PROPORTIONALITY,
                priority=9,
                weight=0.50,
                applicable_domains=["military_ai", "privacy_surveillance", "disaster_response"],
            ),
            EthicalRule(
                rule_id="R20",
                name="Least Restrictive Means",
                description="Choose the least restrictive or invasive option "
                           "that achieves the goal.",
                category=RuleCategory.PROPORTIONALITY,
                priority=9,
                weight=0.48,
            ),
            
            # === BENEFICENCE (Lower Priority - Aspirational) ===
            EthicalRule(
                rule_id="R21",
                name="Maximize Overall Benefit",
                description="When basic ethical constraints are met, choose "
                           "actions that produce the greatest overall benefit.",
                category=RuleCategory.BENEFICENCE,
                priority=10,
                weight=0.45,
            ),
            EthicalRule(
                rule_id="R22",
                name="Long-term Welfare",
                description="Consider long-term consequences and societal welfare, "
                           "not just immediate outcomes.",
                category=RuleCategory.BENEFICENCE,
                priority=10,
                weight=0.42,
            ),
        ]
        
        for rule in all_rules:
            self.rules[rule.rule_id] = rule
    
    def get_rule(self, rule_id: str) -> Optional[EthicalRule]:
        """Get a rule by its ID."""
        return self.rules.get(rule_id)
    
    def get_rules_by_category(self, category: RuleCategory) -> List[EthicalRule]:
        """Get all rules in a given category."""
        return [r for r in self.rules.values() if r.category == category]
    
    def get_rules_by_priority(self, priority: int) -> List[EthicalRule]:
        """Get all rules at a given priority level."""
        return [r for r in self.rules.values() if r.priority == priority]
    
    def get_applicable_rules(self, domain: str) -> List[EthicalRule]:
        """Get all rules applicable to a given domain."""
        return [r for r in self.rules.values() if r.applies_to_domain(domain)]
    
    def get_all_rules_sorted(self) -> List[EthicalRule]:
        """Get all rules sorted by priority (highest first)."""
        return sorted(self.rules.values(), key=lambda r: (r.priority, -r.weight))
    
    def get_rule_hierarchy(self) -> Dict[int, List[EthicalRule]]:
        """Get rules organized by priority level."""
        hierarchy = {}
        for rule in self.rules.values():
            if rule.priority not in hierarchy:
                hierarchy[rule.priority] = []
            hierarchy[rule.priority].append(rule)
        return dict(sorted(hierarchy.items()))
    
    def detect_conflicts(self, triggered_rules: List[str]) -> List[Dict]:
        """
        Detect conflicts between triggered rules.
        Returns a list of conflict descriptions.
        """
        conflicts = []
        rule_objects = [self.rules[rid] for rid in triggered_rules if rid in self.rules]
        
        for i, rule_a in enumerate(rule_objects):
            for rule_b in rule_objects[i+1:]:
                if rule_a.category != rule_b.category:
                    # Different categories can conflict
                    conflicts.append({
                        "rule_a": rule_a.rule_id,
                        "rule_b": rule_b.rule_id,
                        "type": "cross_category",
                        "resolution": "priority",
                        "winner": rule_a.rule_id if rule_a.priority < rule_b.priority 
                                  else rule_b.rule_id if rule_b.priority < rule_a.priority
                                  else "tie",
                    })
        
        return conflicts
    
    def __len__(self):
        return len(self.rules)
    
    def __repr__(self):
        return f"EthicalRuleSystem(rules={len(self.rules)})"
