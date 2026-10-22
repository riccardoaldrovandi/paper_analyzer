"""
Scores judge predictions against the gold claims: accuracy, balanced accuracy,
macro-F1 and Cohen's kappa with 95% bootstrap CIs (no GPU needed).

    python scripts/nli/score_judges.py --claims gold/claims.jsonl \
        --predictions logs/nli_predictions [--json summary.json]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.calibration import evaluate  # noqa: E402


def fmt(v: tuple) -> str:
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="gold/claims.jsonl")
    ap.add_argument("--predictions", default="logs/nli_predictions", help="directory with one <judge>.jsonl per judge")
    ap.add_argument("--json", help="also write the full results here")
    args = ap.parse_args()

    with open(args.claims, encoding="utf-8") as f:
        claims = [json.loads(line) for line in f if line.strip()]
    files = sorted(glob.glob(os.path.join(args.predictions, "*.jsonl")))
    if not files:
        print(f"[!] No prediction files in {args.predictions}")
        return 1

    results = {}
    for path in files:
        judge = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            preds = {r["id"]: r["label"] for r in (json.loads(line) for line in f if line.strip())}
        results[judge] = evaluate(claims, preds)

    print(f"Binary task (SUPPORTED vs not), {len(claims)} claims, mean [95% bootstrap CI]")
    print(f"{'judge':<24} {'answered':>8} {'accuracy':>22} {'bal. acc':>22} {'macro-F1':>22} {'kappa':>22}")
    for judge, r in results.items():
        b = r["binary"]
        print(f"{judge:<24} {r['answered']:>3}/{r['total']:<4} {fmt(b['accuracy']):>22} {fmt(b['balanced_accuracy']):>22} "
              f"{fmt(b['macro_f1']):>22} {fmt(b['kappa']):>22}")
    three = {j: r["three_class"] for j, r in results.items() if "three_class" in r}
    if three:
        print("\n3-class task (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO)")
        print(f"{'judge':<24} {'accuracy':>22} {'macro-F1':>22} {'kappa':>22}")
        for judge, t in three.items():
            print(f"{judge:<24} {fmt(t['accuracy']):>22} {fmt(t['macro_f1']):>22} {fmt(t['kappa']):>22}")
    print("\nBinary accuracy by claim origin (perturbed claims are the easy ones):")
    for judge, r in results.items():
        print(f"  {judge:<24} " + "  ".join(f"{o}={v['binary_accuracy']:.3f} (n={v['n']})" for o, v in r["binary_by_origin"].items()))
    print("\nWrong on (binary):")
    for judge, r in results.items():
        print(f"  {judge}: {', '.join(r['wrong_ids']) or '-'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"\n[+] Full results -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
