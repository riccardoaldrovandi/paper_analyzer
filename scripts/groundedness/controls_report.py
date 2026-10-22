"""
Step (CPU, PC): reads the judge's scores of the control claims and reports the
false-support rate (negative control) and the detection rate per perturbation.

    python scripts/groundedness/controls_report.py --controls logs/groundedness/controls_claims.jsonl \
        --scores logs/groundedness/controls_scores.jsonl --out logs/groundedness/controls_summary.json
"""
import argparse
import json
import statistics
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return ph, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctl = [json.loads(l) for l in open(args.controls, encoding="utf-8") if l.strip()]
    p = {s["id"]: s["p_supported"] for s in (json.loads(l) for l in open(args.scores, encoding="utf-8") if l.strip())}
    rows = defaultdict(list)
    for c in ctl:
        if c["id"] in p:
            rows[(c["kind"], c["provider"])].append((c, p[c["id"]]))
    out = {}
    print("Negative control: claim judged against an UNRELATED paper (right verdict: not supported)\n")
    print(f"{'provider':40s} {'n':>5s} {'false-support p>=0.5':>26s} {'p>=0.85':>22s}")
    for (kind, prov), r in sorted(rows.items()):
        if kind != "negative":
            continue
        n = len(r)
        a = wilson(sum(x >= 0.5 for _, x in r), n); b = wilson(sum(x >= 0.85 for _, x in r), n)
        print(f"{prov:40s} {n:5d} {a[0]:8.3f} [{a[1]:.3f}, {a[2]:.3f}]   {b[0]:6.3f} [{b[1]:.3f}, {b[2]:.3f}]")
        out[f"negative|{prov}"] = {"n": n, "false_support_0.5": a, "false_support_0.85": b}
    print("\nPerturbed VERIFIED claims (right verdict: not supported)\n")
    print(f"{'kind':8s} {'provider':38s} {'n':>5s} {'detected p<0.5':>22s} {'downgraded p<0.85':>22s} {'mean p orig -> pert':>22s}")
    for (kind, prov), r in sorted(rows.items()):
        if kind == "negative":
            continue
        n = len(r)
        a = wilson(sum(x < 0.5 for _, x in r), n); b = wilson(sum(x < 0.85 for _, x in r), n)
        mo = statistics.mean(c["orig_p"] for c, _ in r); mp = statistics.mean(x for _, x in r)
        print(f"{kind:8s} {prov:38s} {n:5d} {a[0]:6.3f} [{a[1]:.3f}, {a[2]:.3f}] {b[0]:6.3f} [{b[1]:.3f}, {b[2]:.3f}]   {mo:.2f} -> {mp:.2f}")
        out[f"{kind}|{prov}"] = {"n": n, "detected_0.5": a, "downgraded_0.85": b, "mean_p_orig": mo, "mean_p_pert": mp}
    print("\nnumber: the deterministic layer (a number absent from the paper -> NUMBER_MISMATCH) detects 100% by construction.")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
