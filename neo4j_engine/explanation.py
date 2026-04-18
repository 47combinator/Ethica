"""
Explanation Generator for Graph-Based Reasoning
================================================
Converts raw graph paths into human-readable explanations
of moral decisions.
"""

from typing import Dict, List, Optional


class GraphExplanationGenerator:
    """
    Generates structured explanations from Neo4j reasoning paths.
    
    Output formats:
      - text:       Human-readable multi-line explanation
      - structured: Dict with sections for programmatic consumption
      - path:       Raw graph path for visualization
    """

    def __init__(self, queries=None):
        """
        Args:
            queries: Optional EthicalGraphQueries for live path lookups.
        """
        self.queries = queries

    # -- Full explanation --------------------------------------------------

    def generate_text_explanation(
        self,
        decision: dict,
        include_conflicts: bool = True,
        include_scores: bool = True,
    ) -> str:
        """
        Generate a complete text explanation from a GraphDecision or
        hybrid decision dict.
        """
        lines = []
        source = decision.get("reasoning_source", "graph")
        scenario_id = decision.get("scenario_id", "?")

        lines.append("=" * 60)
        lines.append(f" ETHICAL REASONING REPORT -- {scenario_id}")
        lines.append(f" Source: {source.replace('_', ' ').title()}")
        lines.append("=" * 60)

        # Chosen action
        chosen = decision.get("chosen_action", decision)
        if isinstance(chosen, dict):
            lines.append(f"\n> Chosen Action: {chosen.get('description', chosen.get('chosen_action_desc', '?'))}")
            score_key = "moral_score" if "moral_score" in chosen else "blended_score"
            lines.append(f"  Score: {chosen.get(score_key, '?')}")
        else:
            lines.append(f"\n> Chosen Action: {decision.get('chosen_action_desc', '?')}")
            lines.append(f"  Moral Score: {decision.get('moral_score', '?')}")

        # All action scores
        if include_scores:
            all_scores = decision.get("all_scores", [])
            if all_scores:
                lines.append(f"\n--- Action Scores ---")
                for s in all_scores:
                    desc = s.get("description", s.get("action_id", "?"))
                    if "blended_score" in s:
                        lines.append(
                            f"  {desc}: blended={s['blended_score']}"
                            f"  (graph={s.get('graph_score', '?')}"
                            f", model={s.get('model1_score', s.get('model4_score', '?'))})"
                        )
                    else:
                        lines.append(
                            f"  {desc}: moral_score={s.get('moral_score', '?')}"
                        )

        # Explanation path
        path = decision.get("explanation_path", [])
        if path:
            lines.append(f"\n--- Reasoning Path ---")
            lines.extend(self._format_path(path))

        # Virtue conflicts
        if include_conflicts:
            conflicts = decision.get("virtue_conflicts", [])
            if conflicts:
                lines.append(f"\n--- Virtue Conflicts ---")
                for c in conflicts:
                    va = c.get("virtue_a", "?")
                    vb = c.get("virtue_b", "?")
                    tension = c.get("tension", "?")
                    lines.append(
                        f"  * {va} <-> {vb}"
                        f"  (tension: {tension})"
                    )

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    # -- Structured output -------------------------------------------------

    def generate_structured_explanation(self, decision: dict) -> Dict:
        """
        Return a structured explanation dict for programmatic use.
        """
        path = decision.get("explanation_path", [])
        
        # Group path entries by principle
        principles_involved = {}
        virtues_involved = set()
        consequence_summary = {}

        for entry in path:
            p = entry.get("principle", "Unknown")
            if p not in principles_involved:
                principles_involved[p] = {
                    "weight": entry.get("principle_weight", 0),
                    "consequences": [],
                    "virtues": [],
                }
            principles_involved[p]["consequences"].append({
                "type": entry.get("consequence_type"),
                "value": entry.get("consequence_value"),
                "influence": entry.get("influence"),
            })
            v = entry.get("virtue")
            if v:
                virtues_involved.add(v)
                if v not in principles_involved[p]["virtues"]:
                    principles_involved[p]["virtues"].append(v)

            ct = entry.get("consequence_type", "?")
            consequence_summary[ct] = entry.get("consequence_value", 0)

        return {
            "scenario_id": decision.get("scenario_id"),
            "chosen_action": decision.get("chosen_action_id",
                             decision.get("chosen_action", {}).get("action_id")),
            "moral_score": decision.get("moral_score",
                           decision.get("chosen_action", {}).get("blended_score")),
            "reasoning_source": decision.get("reasoning_source", "graph"),
            "principles_involved": principles_involved,
            "virtues_involved": sorted(virtues_involved),
            "consequence_summary": consequence_summary,
            "virtue_conflicts": decision.get("virtue_conflicts", []),
            "path_length": len(path),
        }

    # -- Path formatting ---------------------------------------------------

    def _format_path(self, path: List[Dict]) -> List[str]:
        """Format raw path records into readable lines."""
        lines = []
        seen_principles = set()

        for entry in path:
            scenario = entry.get("scenario_title", entry.get("scenario_id", "?"))
            action = entry.get("action_desc", "?")
            ctype = entry.get("consequence_type", "?")
            cval = entry.get("consequence_value", "?")
            principle = entry.get("principle", "?")
            virtue = entry.get("virtue")
            influence = entry.get("influence", "?")
            alignment = entry.get("alignment", "?")

            if principle not in seen_principles:
                seen_principles.add(principle)
                lines.append(f"  Scenario: {scenario}")
                lines.append(f"    -> Action: {action}")
                lines.append(
                    f"       -> Consequence: {ctype} = {cval}"
                    f" (influence: {influence})"
                )
                lines.append(
                    f"          -> Principle: {principle}"
                )
                if virtue:
                    lines.append(
                        f"             -> Virtue: {virtue}"
                        f" (alignment: {alignment})"
                    )
            elif virtue:
                # Additional virtue link for same principle
                lines.append(
                    f"             -> Virtue: {virtue}"
                    f" (alignment: {alignment})"
                )

        return lines

    # -- Quick summary -----------------------------------------------------

    def summarize(self, decision: dict) -> str:
        """One-line summary of the decision."""
        chosen = decision.get("chosen_action", decision)
        desc = (chosen.get("description", "") or
                chosen.get("chosen_action_desc", "?"))
        score = (chosen.get("moral_score", None) or
                 chosen.get("blended_score", "?"))
        source = decision.get("reasoning_source", "graph")
        return f"[{source}] {decision.get('scenario_id', '?')}: {desc} (score={score})"
