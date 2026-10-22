"""
Controls for the groundedness judge, on REAL claims (docs/METRICS.md 7.4).

The gold claims (6.1) are short and synthetic; what the judge does on the long
claims it actually sees is measured here with claims whose right answer is known
by construction:

- negative control: a claim of paper A judged against an UNRELATED paper B. The
  right verdict is "not supported". Share judged supported = false-support rate.
- perturbed claim: a claim the judge called VERIFIED, changed so that it is no
  longer true of its paper. The right verdict is "not supported".
      number   one number (2+ digits) replaced by one that is not in the paper
      flip     one word replaced by its opposite ("increase" -> "decrease")
      negate   a copula negated ("is" -> "is not")
  detection rate = perturbed claims judged < threshold / perturbed claims.

Both are synthetic: some flips are not perfectly meaningful and an unrelated
paper may still support a very generic claim, so the rates are read as a
measurement of sensitivity, not as an error rate on the model's output.
"""
import re

from src.evaluation.metrics import _numbers

# Paper groups that share a topic: a claim is never judged against a paper of
# its own group (twistronics/Lieb lattices; SYK/OTOC/quantum chaos), where a
# "wrong" paper could legitimately support it.
GROUPS = {
    "3056128": "lattice", "3057253": "lattice", "1751397": "lattice",
    "1473712": "chaos", "1519859": "chaos",
}


def group_of(source_file: str) -> str:
    return GROUPS.get(source_file.split("_")[0], source_file.split("_")[0])


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def swap_number(claim: str, paper_text: str):
    """Replace the first number (2+ digits) by 1.5x its value (same decimals) if the result is not in the paper."""
    available = _numbers(paper_text)
    for m in _NUMBER.finditer(claim):
        digits = m.group().replace(".", "")
        if len(digits) < 2:
            continue
        decimals = len(m.group().split(".")[1]) if "." in m.group() else 0
        for factor in (1.5, 2.5, 0.4):
            new = f"{float(m.group()) * factor:.{decimals}f}"
            if new != m.group() and new not in available:
                return claim[:m.start()] + new + claim[m.end():]
    return None


# Longer phrases first so "non-monotonic" is not turned into "non-non-monotonic".
FLIPS = [
    ("non-monotonic", "monotonic"), ("nonmonotonic", "monotonic"),
    ("increases", "decreases"), ("increase", "decrease"), ("increased", "decreased"),
    ("higher", "lower"), ("larger", "smaller"), ("faster", "slower"), ("stronger", "weaker"),
    ("improves", "degrades"), ("outperforms", "underperforms"), ("enhances", "suppresses"),
    ("better", "worse"), ("greater", "smaller"), ("above", "below"), ("positive", "negative"),
    ("supports", "contradicts"), ("exponential", "linear"), ("integrable", "chaotic"),
]
_FLIP_PATTERNS = [(re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE), b) for a, b in FLIPS] + \
                 [(re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE), a) for a, b in FLIPS]


def flip_word(claim: str):
    for pattern, repl in _FLIP_PATTERNS:
        m = pattern.search(claim)
        if m:
            new = repl.capitalize() if m.group()[0].isupper() else repl
            return claim[:m.start()] + new + claim[m.end():]
    return None


_NEGATE = re.compile(r"\b(is|are|was|were|does|do|can|may|shows?|provides?)\b(?! not)")


_ALREADY_NEGATIVE = re.compile(r"\b(not|no|never|neither|nor|without|cannot|n't)\b", re.IGNORECASE)


def negate(claim: str):
    if _ALREADY_NEGATIVE.search(claim):   # "no" / "not" present: negating again would be a double negative
        return None
    m = _NEGATE.search(claim)
    if not m:
        return None
    w = m.group()
    if w in ("is", "are", "was", "were", "can", "may"):
        new = f"{w} not"
    elif w in ("does", "do"):
        new = f"{w} not"
    else:   # shows / provides -> does not show / provide
        new = "does not " + w.rstrip("s")
    return claim[:m.start()] + new + claim[m.end():]
