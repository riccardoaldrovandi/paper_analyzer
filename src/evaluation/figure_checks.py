"""
Figure-reading checks (gold/figure_checks.jsonl, written by scripts/matrix/build_figure_checks.py).

Rubric recall and Level 2 can't see whether a figure was read correctly: a
wrong description that contains the rubric keywords counts as found, and a
right value read off a plot ("levels off at ~0.5") counts as ungrounded because
the paper's text doesn't print it. These checks ask what only a correct reading
answers (values at the start, peaks, asymptotes, which curve decays faster).

For each check the answer's text about that figure is collected: the
"figures" entries whose id or caption names the figure, plus the key_results
and visual findings that cite it. The check passes if an "accept" regex
matches that text and no "reject" regex does (a wrong statement next to a right
one fails). A figure the answer doesn't describe at all fails every check.

in_text marks the checks an "accept" pattern also finds in the paper's text:
those can be answered from the caption or the body. The others ("visual") are
the ones that need the image: they are what the modes and models differ on.
"""
import re
import json

from src.evaluation.metrics import _as_text
from src.evaluation.rubric import _key

_FLAGS = re.IGNORECASE | re.DOTALL


def load_figure_checks(path: str) -> dict:
    """{paper name: [check, ...]} from the JSONL file."""
    checks = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            for field in ("paper", "figure", "id", "accept"):
                if not c.get(field):
                    raise ValueError(f"figure check line {line_no}: missing '{field}'")
            for pattern in c["accept"] + c.get("reject", []):
                re.compile(pattern, _FLAGS)
            checks.setdefault(c["paper"], []).append(c)
    return checks


def checks_for(checks: dict, source_file: str) -> list:
    return checks.get(_key(source_file), [])


def _names_figure(text: str, number: int) -> bool:
    return re.search(rf"\b(?:fig(?:ure)?s?\.?|panel)\s*\(?{number}(?![0-9])", text, re.IGNORECASE) is not None


def figure_text(parsed: dict, number: int) -> str:
    """Everything the answer says about figure <number>."""
    parts = []
    for item in parsed.get("figures") or []:
        if isinstance(item, dict):
            label = f"{item.get('id', '')} {item.get('caption', '')}"
            if _names_figure(label, number) or str(item.get("id", "")).strip() == str(number):
                parts.append(_as_text(item))
    for key in ("key_results", "visual_analysis_findings"):
        for item in parsed.get(key) or []:
            if isinstance(item, dict) and _names_figure(f"{item.get('source', '')} {item.get('figure', '')}", number):
                parts.append(_as_text(item))
    return "\n".join(parts)


def score_figure_checks(parsed: dict, checks: list, paper_text: str) -> dict:
    """Pass/fail of every check on one answer; "visual" counts only the checks not answerable from the text."""
    parsed = parsed if isinstance(parsed, dict) else {}
    results = []
    for c in checks:
        text = figure_text(parsed, c["figure"])
        accepted = any(re.search(p, text, _FLAGS) for p in c["accept"])
        rejected = any(re.search(p, text, _FLAGS) for p in c.get("reject", []))
        in_text = any(re.search(p, paper_text, _FLAGS) for p in c["accept"])
        results.append({"id": c["id"], "figure": c["figure"], "passed": bool(text) and accepted and not rejected,
                        "described": bool(text), "wrong": rejected, "in_text": in_text})
    visual = [r for r in results if not r["in_text"]]
    return {
        "checks": len(results),
        "passed": sum(r["passed"] for r in results),
        "visual_checks": len(visual),
        "visual_passed": sum(r["passed"] for r in visual),
        "wrong": [r["id"] for r in results if r["wrong"]],
        "not_described": sorted({r["figure"] for r in results if not r["described"]}),
        "failed": [r["id"] for r in results if not r["passed"]],
    }
