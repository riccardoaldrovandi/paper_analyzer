"""
Stability of an extraction across runs (seeds): does the model say the same
thing when asked again? No judge and no GPU: it only compares the answers.

For one provider and one paper with runs 1..k, each claim of run i (a sentence
or list item of a text field, groundedness.extract_claims) is matched to the
most similar claim of run j. Similarity is the Dice coefficient of the claims'
content stems (first 5 letters of words of 4+ letters, plus numbers):

    dice(A, B) = 2 |A and B| / (|A| + |B|),      matched if dice >= MATCH_MIN

    claim agreement of run i with j  =  matched claims of i / claims of i
    claim agreement of the paper     =  mean over ordered pairs (i, j), i != j

An agreement near 1 means the answers are the same facts in other words; a low
one means that every run says something different, which makes any single run a
weak basis for a conclusion. It is not correctness: a model can be stably wrong.

    number agreement = Jaccard of the sets of numbers (2+ digits) cited in the
                       answer's text fields, mean over pairs of runs.
"""
import re
from itertools import permutations

from src.evaluation.groundedness import extract_claims
from src.evaluation.metrics import _numbers

MATCH_MIN = 0.5
_WORD = re.compile(r"[^\W\d_]{4,}")


def _stems(text: str) -> frozenset:
    return frozenset({w[:5] for w in _WORD.findall(text.lower())}
                     | {n for n in _numbers(text) if len(n.replace(".", "")) > 1})


def dice(a: frozenset, b: frozenset) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def claim_agreement(runs: list) -> float:
    """runs: list of parsed answers (dicts) of one paper. None if fewer than 2 runs or no claims."""
    sets = [[_stems(c["claim"]) for c in extract_claims(r)] for r in runs]
    scores = []
    for i, j in permutations(range(len(sets)), 2):
        if not sets[i]:
            continue
        matched = sum(max((dice(a, b) for b in sets[j]), default=0.0) >= MATCH_MIN for a in sets[i])
        scores.append(matched / len(sets[i]))
    return sum(scores) / len(scores) if scores else None


def number_agreement(runs: list) -> float:
    sets = []
    for r in runs:
        text = " ".join(c["claim"] for c in extract_claims(r))
        sets.append({n for n in _numbers(text) if len(n.replace(".", "")) > 1})
    scores = [len(a & b) / len(a | b) for a, b in permutations(sets, 2) if a | b]
    return sum(scores) / len(scores) if scores else None
