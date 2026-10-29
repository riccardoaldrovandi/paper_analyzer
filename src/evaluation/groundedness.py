"""
Layered groundedness check of an LLM extraction against its source paper.

Layer 1, judge: the answer's text fields are split into claims (one per
sentence / list item) and a local judge (Bespoke-MiniCheck-7B, see judges.py)
gives each claim a probability p that the paper supports it.
Layer 2, deterministic: every number a claim states must appear in the paper
(the judges are blind to a swapped number, see docs/METRICS.md 6.2).
Layer 3, doubt: a claim whose p is not decisive is reported as DOUBTFUL instead
of being forced into true/false.
Layer 4, audit: the judge is NOT reliable enough on real, long claims to count
errors by itself (docs/METRICS.md 7.2). Its bands only decide where a human
looks first; the actual error rate comes from a stratified audit of a sample of
claims per band, extrapolated to all claims with stratified_estimate().

    UNSUPPORTED      p <  0.5
    NUMBER_MISMATCH  p >= 0.5 but a number of the claim is not in the paper
    DOUBTFUL         0.5 <= p < 0.85
    VERIFIED         p >= 0.85 and all numbers found

    verified rate = VERIFIED / claims,   flagged rate = (UNSUPPORTED + NUMBER_MISMATCH) / claims
    These are judge signals, not error rates: most flagged claims were true in the audit.

This module needs no GPU: the probabilities come from scripts/groundedness.
"""
import random

from src.evaluation.judges import split_sentences
from src.evaluation.metrics import _as_text, _is_empty, _numbers
from src.prompt_v2 import CLAIM_PATHS_V2

# Fields that state something about the paper. Not included: datasets_used
# (names, checked by dataset_grounding), relevant_elements (references, checked
# by reference_grounding) and visual_analysis_findings (AI curve fits, checked by
# visual_separation).
CLAIM_FIELDS = ("problem_addressed", "methodology", "experiments", "results", "limitations",
                "future_work", "main_contributions")
VERIFIED_MIN = 0.85
UNSUPPORTED_MAX = 0.5
MIN_CLAIM_CHARS = 25
STATUSES = ("VERIFIED", "DOUBTFUL", "NUMBER_MISMATCH", "UNSUPPORTED")


def _claim_sources(parsed: dict) -> list:
    """(field name, list of values) per claim field; v2 answers (they have "summary") use CLAIM_PATHS_V2."""
    if "summary" not in parsed:
        return [(f, parsed.get(f)) for f in CLAIM_FIELDS]
    out = []
    for field, sub in CLAIM_PATHS_V2:
        value = parsed.get(field)
        if sub is None:
            out.append((field, value))
        elif isinstance(value, dict):
            out.append((f"{field}.{sub}", value.get(sub)))
        elif isinstance(value, list):
            # A list of objects (key_results, tables, figures): that subfield of each.
            out.append((f"{field}.{sub}", [v.get(sub) for v in value if isinstance(v, dict)]))
    return out


def extract_claims(parsed: dict) -> list:
    """[{"field", "claim"}]: one claim per sentence of a text field, one per list item (split into sentences)."""
    out = []
    for field, value in _claim_sources(parsed):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item is None or _is_empty(item):
                continue
            for sentence in split_sentences(" ".join(_as_text(item).split())):
                if len(sentence) >= MIN_CLAIM_CHARS:
                    out.append({"field": field, "claim": sentence})
    return out


def ungrounded_numbers(claim: str, paper_text: str) -> list:
    """Numbers (2+ digits, like number_grounding) that the claim states and the paper does not contain."""
    cited = {n for n in _numbers(claim) if len(n.replace(".", "")) > 1}
    available = _numbers(paper_text)
    return sorted(n for n in cited if n not in available)


def status(p_supported: float, ungrounded: list) -> str:
    if p_supported < UNSUPPORTED_MAX:
        return "UNSUPPORTED"
    if ungrounded:
        return "NUMBER_MISMATCH"
    if p_supported < VERIFIED_MIN:
        return "DOUBTFUL"
    return "VERIFIED"


def summarize(statuses: list) -> dict:
    n = len(statuses)
    if not n:
        return {"claims": 0, "verified": None, "doubtful": None, "number_mismatch": None, "unsupported": None,
                "flagged": None}
    share = {s.lower(): statuses.count(s) / n for s in STATUSES}
    flagged = share["unsupported"] + share["number_mismatch"]
    return {"claims": n, **share, "flagged": flagged}


def stratified_estimate(strata_sizes: dict, audit: dict, outcomes: set, n_draws: int = 10000, seed: int = 0) -> tuple:
    """
    Share of ALL claims whose audited outcome is in `outcomes`, from an audit of a sample per band.

    strata_sizes: {band: number of claims in that band}.  audit: {band: [outcome of each audited claim]}.
    The estimate is sum_h N_h * p_h / N. Each p_h gets a Jeffreys Beta(k + 1/2, n - k + 1/2) posterior, so a band
    with 0 events in 20 audited claims still carries uncertainty (a plain bootstrap would give it none).
    Returns (point estimate, low, high); the interval is the 95% credible interval of the weighted share.
    """
    total = sum(strata_sizes.values())
    counts = {h: (sum(o in outcomes for o in v), len(v)) for h, v in audit.items()}
    point = sum(strata_sizes[h] * k / n for h, (k, n) in counts.items()) / total
    rng = random.Random(seed)
    draws = sorted(sum(strata_sizes[h] * rng.betavariate(k + 0.5, n - k + 0.5) for h, (k, n) in counts.items()) / total
                   for _ in range(n_draws))
    return point, draws[int(0.025 * n_draws)], draws[int(0.975 * n_draws) - 1]


def stratified_difference(sizes_a: dict, audit_a: dict, sizes_b: dict, audit_b: dict, outcomes: set,
                          n_draws: int = 10000, seed: int = 0) -> tuple:
    """
    Difference (A - B) between two providers in the share of claims with an audited outcome in `outcomes`.
    Each provider's share is stratified_estimate's weighted Beta posterior; the two are independent.
    Returns (difference, low, high, P(A > B)).
    """
    def draws(sizes, audit, rng):
        total = sum(sizes.values())
        counts = {h: (sum(o in outcomes for o in v), len(v)) for h, v in audit.items()}
        return sum(sizes[h] * rng.betavariate(k + 0.5, n - k + 0.5) for h, (k, n) in counts.items()) / total

    rng = random.Random(seed)
    diffs = sorted(draws(sizes_a, audit_a, rng) - draws(sizes_b, audit_b, rng) for _ in range(n_draws))
    point = (stratified_estimate(sizes_a, audit_a, outcomes, 1)[0] - stratified_estimate(sizes_b, audit_b, outcomes, 1)[0])
    return point, diffs[int(0.025 * n_draws)], diffs[int(0.975 * n_draws) - 1], sum(d > 0 for d in diffs) / n_draws
