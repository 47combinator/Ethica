"""
Explanation Generator for Model 1: Rule-Based Moral AI
======================================================

Generates human-readable explanations of moral decisions
for research analysis and transparency evaluation.
"""

from typing import Dict, List, Optional
from .rules import EthicalRuleSystem


class ExplanationGenerator:
    """
    Generates structured, human-readable explanations
    of the moral decision-making process.
    """
    
    def __init__(self, rule_system: Optional[EthicalRuleSystem] = None):
        self.rule_system = rule_system or EthicalRuleSystem()
    
    def generate_full_explanation(self, decision, scenario: Dict) -> str:
        """
        Generate a comprehensive explanation of a moral decision.
        
        Args:
            decision: MoralDecision object
            scenario: Original scenario dictionary
            
        Returns:
            Formatted explanation string
        """
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append(f"MORAL DECISION REPORT")
        lines.append(f"Scenario: {scenario.get('title', decision.scenario_id)}")
        lines.append(f"Category: {scenario.get('category', 'Unknown')}")
        lines.append("=" * 60)
        
        # Scenario Description
        lines.append("")
        lines.append("SCENARIO:")
        lines.append(f"  {scenario.get('description', 'No description')}")
        
        # Ethical Factors
        lines.append("")
        lines.append("IDENTIFIED ETHICAL FACTORS:")
        for factor in decision.ethical_factors:
            intensity = self._intensity_label(factor.value)
            lines.append(f"  * {factor.name}: {intensity} ({factor.value:.2f})")
        
        # Action Evaluation
        lines.append("")
        lines.append("ACTION EVALUATION:")
        for action_score in sorted(decision.action_scores, 
                                    key=lambda x: x.total_score, reverse=True):
            marker = " [CHOSEN]" if action_score.action_id == decision.chosen_action else ""
            lines.append(f"")
            lines.append(f"  [{action_score.action_id}] {action_score.action_description}{marker}")
            lines.append(f"      Moral Score: {action_score.total_score:.4f}")
            
            if action_score.supported_rules:
                rule_names = self._get_rule_names(action_score.supported_rules[:4])
                lines.append(f"      Supports: {', '.join(rule_names)}")
            
            if action_score.violated_rules:
                rule_names = self._get_rule_names(action_score.violated_rules[:4])
                lines.append(f"      Violates: {', '.join(rule_names)}")
        
        # Conflicts
        if decision.conflicts_detected:
            lines.append("")
            lines.append("RULE CONFLICTS DETECTED:")
            for conflict in decision.conflicts_detected[:5]:
                rule_a = self.rule_system.get_rule(conflict["rule_a"])
                rule_b = self.rule_system.get_rule(conflict["rule_b"])
                winner_id = conflict.get("winner", "tie")
                if winner_id != "tie":
                    winner = self.rule_system.get_rule(winner_id)
                    lines.append(
                        f"  * '{rule_a.name}' vs '{rule_b.name}'"
                        f" -> Winner: '{winner.name}' (higher priority)"
                    )
                else:
                    lines.append(
                        f"  * '{rule_a.name}' vs '{rule_b.name}' -> TIE"
                    )
        
        # Decision
        lines.append("")
        lines.append("DECISION:")
        dom_rule = self.rule_system.get_rule(decision.dominant_rule)
        lines.append(f"  Selected Action: {decision.chosen_action_description}")
        lines.append(f"  Confidence: {decision.confidence:.1%}")
        if dom_rule:
            lines.append(f"  Dominant Rule: '{dom_rule.name}' (Priority {dom_rule.priority})")
        
        # Reasoning Chain
        lines.append("")
        lines.append("REASONING CHAIN:")
        for i, step in enumerate(decision.reasoning_chain, 1):
            lines.append(f"  {i}. {step}")
        
        # Summary
        lines.append("")
        lines.append("SUMMARY:")
        summary = self._generate_summary(decision, scenario, dom_rule)
        lines.append(f"  {summary}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_brief_explanation(self, decision, scenario: Dict) -> str:
        """Generate a brief one-paragraph explanation."""
        dom_rule = self.rule_system.get_rule(decision.dominant_rule)
        
        action_desc = decision.chosen_action_description
        confidence = decision.confidence
        
        factors = ", ".join([f.name for f in decision.ethical_factors[:3]])
        
        rule_name = dom_rule.name if dom_rule else "Unknown Rule"
        
        num_conflicts = len(decision.conflicts_detected)
        conflict_text = ""
        if num_conflicts > 0:
            conflict_text = (
                f" Despite {num_conflicts} rule conflict(s) detected, "
                f"the conflict was resolved using rule priority hierarchy."
            )
        
        return (
            f"The system selected '{action_desc}' with {confidence:.0%} confidence. "
            f"Key ethical dimensions considered: {factors}. "
            f"The dominant rule was '{rule_name}' "
            f"(Priority {dom_rule.priority if dom_rule else 'N/A'}).{conflict_text}"
        )
    
    def _generate_summary(self, decision, scenario: Dict, dom_rule) -> str:
        """Generate a contextual summary."""
        category = scenario.get("category", "general")
        action = decision.chosen_action_description
        confidence = decision.confidence
        
        if confidence > 0.7:
            certainty = "high confidence"
        elif confidence > 0.4:
            certainty = "moderate confidence"
        else:
            certainty = "low confidence (indicates moral ambiguity)"
        
        if dom_rule:
            return (
                f"In this {category.replace('_', ' ')} scenario, the system chose "
                f"to '{action}' with {certainty}. The decision was primarily driven by "
                f"the rule '{dom_rule.name}' which has priority level {dom_rule.priority}. "
                f"{len(decision.conflicts_detected)} ethical conflicts were identified "
                f"and resolved through the rule hierarchy."
            )
        return f"Decision: {action} with {certainty}."
    
    def _intensity_label(self, value: float) -> str:
        """Convert a factor intensity to a human label."""
        if value >= 0.8:
            return "Very High"
        elif value >= 0.6:
            return "High"
        elif value >= 0.4:
            return "Medium"
        elif value >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _get_rule_names(self, rule_ids: List[str]) -> List[str]:
        """Get human-readable names for a list of rule IDs."""
        names = []
        for rid in rule_ids:
            rule = self.rule_system.get_rule(rid)
            if rule:
                names.append(f"{rule.name} ({rid})")
        return names
