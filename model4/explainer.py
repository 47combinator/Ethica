"""
Explainer for Model 4: Virtue Ethics Moral AI
==============================================
"""

from typing import Dict


class VirtueExplainer:
    def __init__(self, virtue_system=None):
        from .virtues import VirtueSystem
        self.vs = virtue_system or VirtueSystem()
    
    def generate_full_explanation(self, decision, scenario: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL 4: VIRTUE ETHICS DECISION REPORT")
        lines.append(f"Scenario: {scenario.get('title', decision.scenario_id)}")
        lines.append(f"Category: {scenario.get('category', 'Unknown')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("SCENARIO:")
        lines.append(f"  {scenario.get('description', 'N/A')}")
        
        lines.append("")
        lines.append("CONTEXT ANALYSIS (Phronesis):")
        for sig in decision.context_signals:
            lines.append(f"  * {sig}")
        
        lines.append("")
        lines.append("VIRTUE WEIGHTS (Context-Adjusted):")
        for vid, w in sorted(decision.virtue_weights.items(), key=lambda x: -x[1]):
            v = self.vs.get_virtue(vid)
            name = v.name if v else vid
            bar = "#" * int(w * 15)
            lines.append(f"  {name:15s} {w:.3f} {bar}")
        
        lines.append("")
        lines.append("ACTION EVALUATION:")
        for a in sorted(decision.action_scores, key=lambda x: -x["score"]):
            m = " [CHOSEN]" if a["is_chosen"] else ""
            lines.append(f"  [{a['action_id']}] {a['description']}{m}")
            lines.append(f"      Overall Score: {a['score']:.4f}")
            top_v = sorted(a["virtue_profile"].items(), key=lambda x: -x[1])[:4]
            for vid, vs in top_v:
                v = self.vs.get_virtue(vid)
                vn = v.name if v else vid
                lines.append(f"        {vn}: {vs:.3f}")
        
        if decision.conflicts:
            lines.append("")
            lines.append("VIRTUE CONFLICTS:")
            for c in decision.conflicts[:5]:
                va = self.vs.get_virtue(c["virtue_a"])
                vb = self.vs.get_virtue(c["virtue_b"])
                dom = self.vs.get_virtue(c["dominant"])
                na, nb, nd = va.name if va else c["virtue_a"], vb.name if vb else c["virtue_b"], dom.name if dom else c["dominant"]
                lines.append(f"  * {na} ({c['score_a']:.2f}) vs {nb} ({c['score_b']:.2f})")
                lines.append(f"    Tension: {c['tension']:.2f} -> {nd} prevails")
        
        lines.append("")
        lines.append("DECISION:")
        lines.append(f"  Selected: {decision.chosen_action_description}")
        lines.append(f"  Confidence: {decision.confidence:.1%}")
        lines.append(f"  Dominant Virtue: {decision.dominant_virtue_name}")
        lines.append(f"  Human Agreement: {decision.human_agreement:.1%}")
        
        lines.append("")
        lines.append("VIRTUE JUSTIFICATION:")
        dom = self.vs.get_virtue(decision.dominant_virtue)
        desc = dom.description if dom else "moral reasoning"
        lines.append(f"  This action was chosen because it best expresses the")
        lines.append(f"  virtue of {decision.dominant_virtue_name} ({desc}).")
        lines.append(f"  Context analysis adjusted virtue weights to prioritize")
        lines.append(f"  what matters most in this {scenario.get('category', 'ethical')} scenario.")
        
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def generate_brief_explanation(self, decision, scenario: Dict) -> str:
        return (
            f"Through virtue ethics analysis, the model chose "
            f"'{decision.chosen_action_description}' with {decision.confidence:.0%} confidence. "
            f"Dominant virtue: {decision.dominant_virtue_name}. "
            f"{len(decision.conflicts)} virtue conflicts resolved. "
            f"Human agreement: {decision.human_agreement:.0%}."
        )
