"""
Explainer for Model 3: RLHF Moral AI
=====================================
"""

from typing import Dict


class RLHFExplainer:
    def generate_full_explanation(self, decision, scenario: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL 3: RLHF MORAL DECISION REPORT")
        lines.append(f"Scenario: {scenario.get('title', decision.scenario_id)}")
        lines.append(f"Category: {scenario.get('category', 'Unknown')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("SCENARIO:")
        lines.append(f"  {scenario.get('description', 'N/A')}")
        lines.append("")
        lines.append("ALIGNMENT PATTERN:")
        lines.append(f"  {decision.learned_pattern}")
        lines.append("")
        lines.append("ACTION SCORES (Policy + Reward):")
        for a in sorted(decision.action_scores, key=lambda x: -x["policy_score"]):
            m = " [CHOSEN]" if a["is_chosen"] else ""
            lines.append(f"  [{a['action_id']}] {a['description']}{m}")
            lines.append(f"      Policy: {a['policy_score']:.4f} | Reward: {a['reward_score']:.4f}")
        lines.append("")
        lines.append("RLHF ANALYSIS:")
        lines.append(f"  Confidence: {decision.confidence:.1%}")
        lines.append(f"  Human Agreement: {decision.human_agreement:.1%}")
        lines.append(f"  Avg Reward Score: {decision.avg_reward:.4f}")
        lines.append(f"  Sycophancy Risk: {decision.sycophancy_risk:.1%}")
        if decision.sycophancy_risk > 0.5:
            lines.append("  WARNING: High sycophancy detected - model may be")
            lines.append("  optimizing for approval rather than ethical correctness.")
        lines.append("")
        lines.append("EXPLANATION:")
        lines.append(f"  Through reinforcement learning from human feedback,")
        lines.append(f"  the model learned that '{decision.learned_pattern}'")
        lines.append(f"  is the preferred moral reasoning in this context.")
        lines.append(f"  The reward model confirmed this with score {decision.avg_reward:.3f}.")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_brief_explanation(self, decision, scenario: Dict) -> str:
        return (
            f"RLHF model selected '{decision.chosen_action_description}' "
            f"with {decision.confidence:.0%} confidence. "
            f"Pattern: {decision.learned_pattern}. "
            f"Human agreement: {decision.human_agreement:.0%}. "
            f"Sycophancy risk: {decision.sycophancy_risk:.0%}."
        )
