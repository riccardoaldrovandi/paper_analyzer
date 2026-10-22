"""
Agreement between a groundedness judge and the gold claims (gold/claims.jsonl).

Gold labels are SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO. A 3-class judge
(NLI) is scored on all three; a binary judge (MiniCheck: supported or not) only
says SUPPORTED / NOT_SUPPORTED, so every judge is also scored on the binary
task "SUPPORTED vs everything else", the one the Groundedness score needs.

    accuracy     = share of claims where judge == gold
    macro-F1     = mean over classes of 2PR/(P+R)
    kappa        = (p_o - p_e) / (1 - p_e),  p_e = sum_c gold_c * judge_c / n^2
    balanced acc = mean over classes of recall

Every metric gets a 95% percentile bootstrap over claims.
"""
import random
from collections import Counter

THREE_CLASS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
BINARY = ("SUPPORTED", "NOT_SUPPORTED")
N_BOOTSTRAP = 2000
SEED = 0


def to_binary(label: str) -> str:
    return "SUPPORTED" if label == "SUPPORTED" else "NOT_SUPPORTED"


def accuracy(gold: list, pred: list) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def _per_class(gold: list, pred: list, labels: tuple) -> dict:
    out = {}
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: list, pred: list, labels: tuple) -> float:
    per = _per_class(gold, pred, labels)
    return sum(v["f1"] for v in per.values()) / len(labels)


def balanced_accuracy(gold: list, pred: list, labels: tuple) -> float:
    per = _per_class(gold, pred, labels)
    present = [v["recall"] for v in per.values() if v["support"]]
    return sum(present) / len(present)


def cohen_kappa(gold: list, pred: list) -> float:
    n = len(gold)
    p_o = accuracy(gold, pred)
    g, p = Counter(gold), Counter(pred)
    p_e = sum(g[c] * p[c] for c in set(g) | set(p)) / (n * n)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def confusion(gold: list, pred: list, labels: tuple) -> dict:
    """{gold label: {predicted label: count}}."""
    table = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        table[g][p] = table[g].get(p, 0) + 1
    return table


def bootstrap(metric, gold: list, pred: list, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
    """(value, low, high): the metric and its 95% percentile bootstrap over claims."""
    rng = random.Random(seed)
    n = len(gold)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(metric([gold[i] for i in idx], [pred[i] for i in idx]))
    values.sort()
    return metric(gold, pred), values[int(0.025 * n_boot)], values[int(0.975 * n_boot) - 1]


def _summary(gold: list, pred: list, labels: tuple, n_boot: int) -> dict:
    return {
        "n": len(gold),
        "accuracy": bootstrap(accuracy, gold, pred, n_boot),
        "balanced_accuracy": bootstrap(lambda g, p: balanced_accuracy(g, p, labels), gold, pred, n_boot),
        "macro_f1": bootstrap(lambda g, p: macro_f1(g, p, labels), gold, pred, n_boot),
        "kappa": bootstrap(cohen_kappa, gold, pred, n_boot),
        "per_class": _per_class(gold, pred, labels),
        "confusion": confusion(gold, pred, labels),
    }


def evaluate(claims: list, predictions: dict, n_boot: int = N_BOOTSTRAP) -> dict:
    """
    claims: gold records (id, label, origin). predictions: {claim id: judge label}.
    Claims the judge did not answer are counted as errors, not dropped, so a judge
    that crashed halfway cannot look better than one that answered everything.
    Returns the binary summary, the 3-class summary if the judge outputs 3-class
    labels, and the binary accuracy per claim origin.
    """
    ids = [c["id"] for c in claims]
    missing = [i for i in ids if i not in predictions]
    gold3 = [c["label"] for c in claims]
    pred3 = [predictions.get(c["id"], "MISSING") for c in claims]
    gold2 = [to_binary(g) for g in gold3]
    pred2 = [to_binary(p) if p != "MISSING" else "MISSING" for p in pred3]

    result = {"answered": len(ids) - len(missing), "total": len(ids), "binary": _summary(gold2, pred2, BINARY, n_boot)}
    if any(p in ("CONTRADICTED", "NOT_ENOUGH_INFO") for p in pred3):
        result["three_class"] = _summary(gold3, pred3, THREE_CLASS, n_boot)
    by_origin = {}
    for origin in sorted({c.get("origin", "?") for c in claims}):
        sel = [i for i, c in enumerate(claims) if c.get("origin", "?") == origin]
        by_origin[origin] = {"n": len(sel), "binary_accuracy": accuracy([gold2[i] for i in sel], [pred2[i] for i in sel])}
    result["binary_by_origin"] = by_origin
    result["wrong_ids"] = [c["id"] for c, g, p in zip(claims, gold2, pred2) if g != p]
    return result
