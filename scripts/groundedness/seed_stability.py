"""
Step (CPU, PC): run-to-run stability of each provider (src/evaluation/consistency.py)
and a table of cost / reliability per provider.

    python scripts/groundedness/seed_stability.py --records-dir logs_cineca/qwen38 \
        --eval-report logs_cineca/qwen38/eval_report.jsonl --out logs/stability.json

Reads every export-dataset JSONL (*.jsonl, not *.rejected.jsonl) in --records-dir.
Rejected answers (the file next to each run) count as failures in the cost table.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.consistency import claim_agreement, number_agreement  # noqa: E402
from src.evaluation.groundedness import extract_claims  # noqa: E402
from src.evaluation.stats import N_BOOTSTRAP, bootstrap_ci  # noqa: E402

SKIP = ("WRONG", "think16k", "finetune_dataset")


def mean_ci(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return "n/a"
    m, lo, hi = bootstrap_ci(values, N_BOOTSTRAP)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", default="logs/stability.json")
    args = ap.parse_args()

    answers = defaultdict(lambda: defaultdict(list))   # provider -> paper -> [parsed answer per run]
    cost = defaultdict(lambda: {"answers": 0, "rejected": 0, "chars": [], "claims": []})
    for path in sorted(glob.glob(os.path.join(args.records_dir, "*.jsonl"))):
        name = os.path.basename(path)
        if any(s in name for s in SKIP) or name.startswith("eval_report"):
            continue
        rejected = name.endswith(".rejected.jsonl")
        # A rejected record has no provider: it is the one of the run's own file.
        sibling = path.replace(".rejected.jsonl", ".jsonl")
        run_provider = None
        if os.path.exists(sibling):
            with open(sibling, encoding="utf-8") as g:
                first = g.readline()
                run_provider = json.loads(first)["provider"] if first.strip() else None
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                p = r.get("provider") or run_provider
                if p is None:
                    continue
                if rejected:
                    cost[p]["rejected"] += 1
                    continue
                try:
                    parsed = json.loads(r["messages"][-1]["content"])
                except (KeyError, ValueError):
                    cost[p]["rejected"] += 1
                    continue
                answers[p][os.path.basename(r["source_file"])].append(parsed)
                cost[p]["answers"] += 1
                cost[p]["chars"].append(len(r["messages"][-1]["content"]))
                cost[p]["claims"].append(len(extract_claims(parsed)))

    print("Run-to-run stability, mean over papers [95% bootstrap CI over papers]\n")
    print(f"{'provider':42s} {'runs/paper':>10s} {'claim agreement':>26s} {'number agreement':>26s}")
    out = {}
    for p in sorted(answers):
        ca = [claim_agreement(runs) for runs in answers[p].values()]
        na = [number_agreement(runs) for runs in answers[p].values()]
        runs = statistics.mean(len(v) for v in answers[p].values())
        print(f"{p:42s} {runs:10.1f} {mean_ci(ca):>26s} {mean_ci(na):>26s}")
        out[p] = {"claim_agreement": [v for v in ca if v is not None], "number_agreement": [v for v in na if v is not None]}

    print("\nCost / reliability (all runs)\n")
    print(f"{'provider':42s} {'answers':>8s} {'rejected':>9s} {'reject %':>9s} {'chars/answer':>13s} {'claims/answer':>14s}")
    for p in sorted(cost):
        c = cost[p]
        tot = c["answers"] + c["rejected"]
        print(f"{p:42s} {c['answers']:8d} {c['rejected']:9d} {100 * c['rejected'] / max(tot, 1):8.1f}% "
              f"{statistics.mean(c['chars']) if c['chars'] else 0:13.0f} {statistics.mean(c['claims']) if c['claims'] else 0:14.1f}")
        out.setdefault(p, {})["cost"] = {k: v if not isinstance(v, list) else (statistics.mean(v) if v else None) for k, v in c.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[+] {args.out}")


if __name__ == "__main__":
    main()
