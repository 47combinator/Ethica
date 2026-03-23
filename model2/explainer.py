"""
Explanation System for Model 2: Learning-Based Moral AI
=======================================================
Generates explanations for learned moral decisions.
"""

from typing import Dict, List


class Model2Explainer:
    """Generates explanations for Model 2 decisions."""
    
    def generate_full_explanation(self, decision, scenario: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL 2: LEARNED MORAL DECISION REPORT")
        lines.append(f"Scenario: {scenario.get('title', decision.scenario_id)}")
        lines.append(f"Category: {scenario.get('category', 'Unknown')}")
        lines.append("=" * 60)
        
        lines.append("")
        lines.append("SCENARIO:")
        lines.append(f"  {scenario.get('description', 'N/A')}")
        
        lines.append("")
        lines.append("LEARNED PATTERN:")
        lines.append(f"  The model identified: {decision.learned_pattern}")
        
        lines.append("")
        lines.append("ACTION SCORES (from trained model):")
        for a in sorted(decision.action_scores, key=lambda x: -x["score"]):
            marker = " [CHOSEN]" if a["is_chosen"] else ""
            lines.append(f"  [{a['action_id']}] {a['description']}{marker}")
            lines.append(f"      Preference Score: {a['score']:.4f}")
        
        lines.append("")
        lines.append("TOP FEATURE INFLUENCES:")
        for feat, imp in list(decision.feature_influences.items())[:8]:
            bar = "#" * int(imp * 100)
            lines.append(f"  {feat:30s} {imp:.4f} {bar}")
        
        lines.append("")
        lines.append("DECISION:")
        lines.append(f"  Selected: {decision.chosen_action_description}")
        lines.append(f"  Confidence: {decision.confidence:.1%}")
        lines.append(f"  Human Agreement: {decision.human_agreement:.1%}")
        
        lines.append("")
        lines.append("EXPLANATION:")
        lines.append(f"  Based on patterns learned from human moral judgments,")
        lines.append(f"  the model predicts that actions aligned with")
        lines.append(f"  '{decision.learned_pattern}' are preferred in this context.")
        if decision.human_agreement > 0.5:
            lines.append(f"  This aligns with human majority opinion.")
        else:
            lines.append(f"  NOTE: This disagrees with human majority opinion,")
            lines.append(f"  suggesting the model may have learned differently.")
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_brief_explanation(self, decision, scenario: Dict) -> str:
        return (
            f"The learned model selected '{decision.chosen_action_description}' "
            f"with {decision.confidence:.0%} confidence. "
            f"Primary learned pattern: {decision.learned_pattern}. "
            f"Human agreement: {decision.human_agreement:.0%}."
        )
