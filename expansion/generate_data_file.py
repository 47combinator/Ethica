"""
Generate cat_expanded.py from the merged JSON output.
Integrates new scenarios directly into the project's data/ directory.
"""
import json
import os
from collections import defaultdict


def format_value(v):
    """Format a Python value for source code."""
    if isinstance(v, str):
        return '"%s"' % v.replace('"', '\\"')
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, list):
        items = ", ".join(format_value(x) for x in v)
        return "[%s]" % items
    elif isinstance(v, dict):
        items = ", ".join('"%s": %s' % (k, format_value(val)) for k, val in v.items())
        return "{%s}" % items
    else:
        return repr(v)


def format_scenario(s):
    """Format a single scenario as indented Python dict."""
    lines = []
    lines.append("    {")
    lines.append('        "id": "%s", "category": "%s",' % (s["id"], s["category"]))
    lines.append('        "title": %s,' % format_value(s["title"]))
    lines.append('        "description": %s,' % format_value(s["description"]))

    dims = s.get("ethical_dimensions", [])
    dims_str = ", ".join('"%s"' % d for d in dims)
    lines.append('        "ethical_dimensions": [%s],' % dims_str)
    lines.append('        "actions": [')

    for a in s.get("actions", []):
        desc = a["description"].replace('"', '\\"')
        lines.append('            {"id": "%s", "description": "%s",' % (a["id"], desc))
        cons = a.get("consequences", {})
        cons_str = ", ".join('"%s": %s' % (k, v) for k, v in cons.items())
        lines.append('             "consequences": {%s}},' % cons_str)

    lines.append("        ],")
    lines.append("    },")
    return "\n".join(lines)


def main():
    merged_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output", "amr_merged.json"
    )

    with open(merged_path, encoding="utf-8") as f:
        all_s = json.load(f)

    # Only new scenarios
    new = [s for s in all_s if s["id"].startswith(("MM_", "SD_", "SA_"))]

    # Remove source metadata
    for s in new:
        s.pop("source", None)

    # Group by category
    by_cat = defaultdict(list)
    for s in new:
        by_cat[s["category"]].append(s)

    # Build output
    out = []
    out.append('"""')
    out.append("AMR-1000 Expansion: Converted scenarios from experimental datasets.")
    out.append("These %d scenarios supplement the original AMR-220 dataset." % len(new))
    out.append("")
    out.append("Sources:")
    out.append("  - Moral Machine LLM Experiment (GPT-4, GPT-3.5, PaLM 2, Llama 2)")
    out.append("  - Allen AI Scruples Dilemmas (crowd-annotated ethical comparisons)")
    out.append("  - Reddit AITA Anecdotes (community moral judgments)")
    out.append('"""')
    out.append("")

    for cat in sorted(by_cat.keys()):
        scenarios = by_cat[cat]
        var_name = cat.upper() + "_EXPANDED"
        out.append("")
        out.append("# %s - %d new scenarios" % (cat, len(scenarios)))
        out.append("%s = [" % var_name)
        for s in scenarios:
            out.append(format_scenario(s))
        out.append("]")
        out.append("")

    # Write
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "cat_expanded.py"
    )
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print("Written %d scenarios to %s" % (len(new), data_path))
    print("Categories: %s" % ", ".join(sorted(by_cat.keys())))
    print("Counts: %s" % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(by_cat.items())))


if __name__ == "__main__":
    main()
