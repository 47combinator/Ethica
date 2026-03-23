"""
Moral Decision Engine for Model 1: Rule-Based Moral AI
======================================================

The core reasoning system that evaluates ethical scenarios
against the rule system and produces moral decisions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .rules import EthicalRuleSystem, EthicalRule, RuleCategory
import math


@dataclass
class EthicalFactor:
    """Represents an extracted ethical factor from a scenario."""
    name: str
    value: float        # 0.0 to 1.0 intensity
    description: str
    related_rules: List[str] = field(default_factory=list)


@dataclass
class ActionScore:
    """Score for a single action in a scenario."""
    action_id: str
    action_description: str
    total_score: float
    rule_scores: Dict[str, float]      # rule_id -> score
    triggered_rules: List[str]
    violated_rules: List[str]
    supported_rules: List[str]


@dataclass
class MoralDecision:
    """The final moral decision output."""
    scenario_id: str
    chosen_action: str
    chosen_action_description: str
    confidence: float               # 0.0 to 1.0
    action_scores: List[ActionScore]
    conflicts_detected: List[Dict]
    dominant_rule: str
    reasoning_chain: List[str]
    ethical_factors: List[EthicalFactor]


# Mapping from ethical dimensions to rules
DIMENSION_RULE_MAP = {
    "harm": ["R03", "R04", "R05"],
    "life_preservation": ["R01", "R02"],
    "fairness": ["R06", "R07", "R08"],
    "autonomy": ["R11", "R12"],
    "honesty": ["R13", "R14"],
    "privacy": ["R15", "R16"],
    "responsibility": ["R09", "R10"],
    "human_oversight": ["R17", "R18"],
    "proportionality": ["R19", "R20"],
    "beneficence": ["R21", "R22"],
    "discrimination": ["R06", "R07"],
    "deception": ["R13", "R14"],
    "legal_compliance": ["R09", "R20"],
    "transparency": ["R14", "R10"],
    "consent": ["R11", "R12"],
    "manipulation": ["R13", "R11"],
    "safety": ["R01", "R03", "R09"],
    "welfare": ["R21", "R22", "R03"],
}


class MoralDecisionEngine:
    """
    The core moral reasoning engine.
    
    Takes structured ethical scenarios, evaluates them against
    the rule system, resolves conflicts, and produces decisions.
    """
    
    def __init__(self, rule_system: Optional[EthicalRuleSystem] = None):
        self.rule_system = rule_system or EthicalRuleSystem()
        self.decision_log: List[MoralDecision] = []
    
    def evaluate_scenario(self, scenario: Dict) -> MoralDecision:
        """
        Evaluate an ethical scenario and produce a moral decision.
        
        Args:
            scenario: Structured scenario dictionary with:
                - id, category, title, description
                - actions: list of possible actions with consequences
                - ethical_dimensions: list of relevant ethical dimensions
        
        Returns:
            MoralDecision with the chosen action and reasoning
        """
        scenario_id = scenario["id"]
        category = scenario["category"]
        dimensions = scenario.get("ethical_dimensions", [])
        actions = scenario.get("actions", [])
        
        # Step 1: Extract ethical factors
        ethical_factors = self._extract_ethical_factors(scenario)
        
        # Step 2: Get applicable rules
        applicable_rules = self.rule_system.get_applicable_rules(category)
        
        # Step 3: Score each action
        action_scores = []
        for action in actions:
            score = self._score_action(action, applicable_rules, ethical_factors, category)
            action_scores.append(score)
        
        # Step 4: Detect conflicts
        all_triggered = set()
        for score in action_scores:
            all_triggered.update(score.triggered_rules)
        conflicts = self.rule_system.detect_conflicts(list(all_triggered))
        
        # Step 5: Resolve and choose best action
        best_action = max(action_scores, key=lambda s: s.total_score)
        
        # Step 6: Calculate confidence
        scores = [s.total_score for s in action_scores]
        confidence = self._calculate_confidence(scores)
        
        # Step 7: Determine dominant rule
        dominant_rule = self._find_dominant_rule(best_action)
        
        # Step 8: Build reasoning chain
        reasoning = self._build_reasoning_chain(
            scenario, best_action, ethical_factors, conflicts, dominant_rule
        )
        
        decision = MoralDecision(
            scenario_id=scenario_id,
            chosen_action=best_action.action_id,
            chosen_action_description=best_action.action_description,
            confidence=confidence,
            action_scores=action_scores,
            conflicts_detected=conflicts,
            dominant_rule=dominant_rule,
            reasoning_chain=reasoning,
            ethical_factors=ethical_factors,
        )
        
        self.decision_log.append(decision)
        return decision
    
    def _extract_ethical_factors(self, scenario: Dict) -> List[EthicalFactor]:
        """Extract ethical factors from a scenario."""
        factors = []
        dimensions = scenario.get("ethical_dimensions", [])
        actions = scenario.get("actions", [])
        
        # Analyze consequences across all actions to determine factor intensities
        for dim in dimensions:
            # Calculate average intensity of this dimension across actions
            intensities = []
            for action in actions:
                consequences = action.get("consequences", {})
                # Map dimension to consequence keys
                intensity = self._get_dimension_intensity(dim, consequences)
                intensities.append(intensity)
            
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0.5
            related_rules = DIMENSION_RULE_MAP.get(dim, [])
            
            factors.append(EthicalFactor(
                name=dim,
                value=avg_intensity,
                description=f"Ethical dimension '{dim}' detected with intensity {avg_intensity:.2f}",
                related_rules=related_rules,
            ))
        
        return factors
    
    def _get_dimension_intensity(self, dimension: str, consequences: Dict) -> float:
        """Map a dimension name to the relevant consequence intensity."""
        mapping = {
            "harm": "harm_to_others",
            "life_preservation": "lives_at_risk_score",
            "fairness": "fairness_impact",
            "autonomy": "autonomy_impact",
            "honesty": "deception_level",
            "privacy": "privacy_impact",
            "responsibility": "responsibility_score",
            "human_oversight": "oversight_needed",
            "proportionality": "proportionality_score",
            "beneficence": "benefit_score",
            "discrimination": "discrimination_level",
            "deception": "deception_level",
            "legal_compliance": "legal_violation_score",
            "transparency": "transparency_score",
            "consent": "consent_violation",
            "manipulation": "manipulation_level",
            "safety": "safety_risk",
            "welfare": "welfare_impact",
        }
        
        key = mapping.get(dimension, dimension)
        return consequences.get(key, 0.5)
    
    def _score_action(
        self, 
        action: Dict, 
        applicable_rules: List[EthicalRule],
        ethical_factors: List[EthicalFactor],
        domain: str
    ) -> ActionScore:
        """Score an action against all applicable rules."""
        consequences = action.get("consequences", {})
        rule_scores = {}
        triggered_rules = []
        violated_rules = []
        supported_rules = []
        
        for rule in applicable_rules:
            score = self._evaluate_rule_for_action(rule, consequences, ethical_factors)
            rule_scores[rule.rule_id] = score
            
            if abs(score) > 0.1:
                triggered_rules.append(rule.rule_id)
            if score < -0.2:
                violated_rules.append(rule.rule_id)
            if score > 0.2:
                supported_rules.append(rule.rule_id)
        
        # Calculate weighted total score
        total_score = 0.0
        total_weight = 0.0
        for rule in applicable_rules:
            weighted = rule_scores[rule.rule_id] * rule.weight
            # Apply priority multiplier (higher priority rules count more)
            priority_multiplier = 1.0 / math.sqrt(rule.priority)
            total_score += weighted * priority_multiplier
            total_weight += rule.weight * priority_multiplier
        
        if total_weight > 0:
            total_score = total_score / total_weight
        
        return ActionScore(
            action_id=action["id"],
            action_description=action["description"],
            total_score=total_score,
            rule_scores=rule_scores,
            triggered_rules=triggered_rules,
            violated_rules=violated_rules,
            supported_rules=supported_rules,
        )
    
    def _evaluate_rule_for_action(
        self, 
        rule: EthicalRule, 
        consequences: Dict,
        ethical_factors: List[EthicalFactor]
    ) -> float:
        """
        Evaluate how well an action satisfies a specific rule.
        Returns a score from -1.0 (violates) to 1.0 (supports).
        """
        category = rule.category
        
        if category == RuleCategory.LIFE_PRESERVATION:
            lives_risk = consequences.get("lives_at_risk_score", 0.5)
            harm = consequences.get("harm_to_others", 0.5)
            # Lower risk and harm = better score
            return 1.0 - (lives_risk * 0.6 + harm * 0.4) * 2
        
        elif category == RuleCategory.HARM_AVOIDANCE:
            harm_others = consequences.get("harm_to_others", 0.5)
            harm_self = consequences.get("harm_to_self", 0.3)
            collateral = consequences.get("collateral_damage", 0.3)
            total_harm = harm_others * 0.5 + harm_self * 0.2 + collateral * 0.3
            return 1.0 - total_harm * 2
        
        elif category == RuleCategory.FAIRNESS:
            fairness = consequences.get("fairness_impact", 0.5)
            discrimination = consequences.get("discrimination_level", 0.3)
            # Higher fairness = better, higher discrimination = worse
            return fairness - discrimination
        
        elif category == RuleCategory.AUTONOMY:
            autonomy = consequences.get("autonomy_impact", 0.5)
            consent = 1.0 - consequences.get("consent_violation", 0.3)
            return (autonomy + consent) / 2 * 2 - 1
        
        elif category == RuleCategory.HONESTY:
            deception = consequences.get("deception_level", 0.3)
            transparency = consequences.get("transparency_score", 0.5)
            return transparency - deception
        
        elif category == RuleCategory.PRIVACY:
            privacy = 1.0 - consequences.get("privacy_impact", 0.3)
            data_min = 1.0 - consequences.get("data_exposure", 0.3)
            return (privacy + data_min) / 2 * 2 - 1
        
        elif category == RuleCategory.RESPONSIBILITY:
            safety = 1.0 - consequences.get("safety_risk", 0.3)
            accountability = consequences.get("accountability_score", 0.5)
            return (safety + accountability) / 2 * 2 - 1
        
        elif category == RuleCategory.HUMAN_OVERSIGHT:
            oversight = consequences.get("human_oversight_maintained", 0.5)
            escalation = consequences.get("escalation_possible", 0.5)
            return (oversight + escalation) / 2 * 2 - 1
        
        elif category == RuleCategory.PROPORTIONALITY:
            proportional = consequences.get("proportionality_score", 0.5)
            restrictive = 1.0 - consequences.get("restrictiveness", 0.5)
            return (proportional + restrictive) / 2 * 2 - 1
        
        elif category == RuleCategory.BENEFICENCE:
            benefit = consequences.get("benefit_score", 0.5)
            welfare = consequences.get("welfare_impact", 0.5)
            return (benefit + welfare) / 2 * 2 - 1
        
        return 0.0
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """
        Calculate confidence in the decision.
        Higher difference between top scores = higher confidence.
        """
        if len(scores) < 2:
            return 1.0
        
        sorted_scores = sorted(scores, reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        
        # Normalize gap to 0-1 range
        # Larger gaps = more confident
        confidence = min(1.0, max(0.1, gap * 2 + 0.3))
        return round(confidence, 3)
    
    def _find_dominant_rule(self, best_action: ActionScore) -> str:
        """Find the rule that most influenced the decision."""
        if not best_action.rule_scores:
            return "R01"
        
        # Find rule with highest absolute influence
        dominant = max(
            best_action.rule_scores.items(),
            key=lambda x: abs(x[1]) * self.rule_system.get_rule(x[0]).weight
        )
        return dominant[0]
    
    def _build_reasoning_chain(
        self,
        scenario: Dict,
        best_action: ActionScore,
        factors: List[EthicalFactor],
        conflicts: List[Dict],
        dominant_rule: str
    ) -> List[str]:
        """Build a human-readable reasoning chain."""
        chain = []
        
        # Step 1: Identify the situation
        chain.append(
            f"Analyzing scenario: {scenario.get('title', scenario['id'])}"
        )
        
        # Step 2: List identified factors
        factor_names = [f.name for f in factors]
        chain.append(
            f"Identified ethical dimensions: {', '.join(factor_names)}"
        )
        
        # Step 3: Report triggered rules
        if best_action.supported_rules:
            rule_names = []
            for rid in best_action.supported_rules[:5]:
                rule = self.rule_system.get_rule(rid)
                if rule:
                    rule_names.append(f"'{rule.name}' ({rid})")
            chain.append(f"Rules supporting this action: {', '.join(rule_names)}")
        
        # Step 4: Report violated rules
        if best_action.violated_rules:
            rule_names = []
            for rid in best_action.violated_rules[:3]:
                rule = self.rule_system.get_rule(rid)
                if rule:
                    rule_names.append(f"'{rule.name}' ({rid})")
            chain.append(f"Rules potentially violated: {', '.join(rule_names)}")
        
        # Step 5: Report conflicts
        if conflicts:
            chain.append(f"Detected {len(conflicts)} rule conflict(s)")
            for conf in conflicts[:3]:
                rule_a = self.rule_system.get_rule(conf["rule_a"])
                rule_b = self.rule_system.get_rule(conf["rule_b"])
                winner = self.rule_system.get_rule(conf["winner"]) if conf["winner"] != "tie" else None
                if winner:
                    chain.append(
                        f"  Conflict: '{rule_a.name}' vs '{rule_b.name}' "
                        f"-> Resolved by priority: '{winner.name}' wins"
                    )
                else:
                    chain.append(
                        f"  Conflict: '{rule_a.name}' vs '{rule_b.name}' -> Tie"
                    )
        
        # Step 6: State dominant rule
        dom_rule = self.rule_system.get_rule(dominant_rule)
        if dom_rule:
            chain.append(
                f"Dominant rule: '{dom_rule.name}' (Priority {dom_rule.priority}, "
                f"Weight {dom_rule.weight})"
            )
        
        # Step 7: State decision
        chain.append(
            f"Decision: {best_action.action_description} "
            f"(Score: {best_action.total_score:.3f})"
        )
        
        return chain
    
    def batch_evaluate(self, scenarios: List[Dict]) -> List[MoralDecision]:
        """Evaluate a batch of scenarios."""
        decisions = []
        for scenario in scenarios:
            decision = self.evaluate_scenario(scenario)
            decisions.append(decision)
        return decisions
    
    def get_decision_log(self) -> List[MoralDecision]:
        """Get the complete decision log."""
        return self.decision_log
    
    def clear_log(self):
        """Clear the decision log."""
        self.decision_log = []
    
    def get_statistics(self) -> Dict:
        """Get statistics about decisions made so far."""
        if not self.decision_log:
            return {"total_decisions": 0}
        
        total = len(self.decision_log)
        avg_confidence = sum(d.confidence for d in self.decision_log) / total
        
        # Count conflict frequency
        total_conflicts = sum(len(d.conflicts_detected) for d in self.decision_log)
        
        # Most common dominant rules
        rule_counts = {}
        for d in self.decision_log:
            rule_counts[d.dominant_rule] = rule_counts.get(d.dominant_rule, 0) + 1
        
        return {
            "total_decisions": total,
            "average_confidence": round(avg_confidence, 3),
            "total_conflicts": total_conflicts,
            "avg_conflicts_per_decision": round(total_conflicts / total, 2),
            "dominant_rule_distribution": rule_counts,
        }
