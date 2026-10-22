"""
Step 1 (CPU, run on the PC): splits the answers of one or more export-dataset
runs into claims. Writes one line per claim.

    python scripts/groundedness/extract_claims.py --records-dir logs_cineca/qwen38 \
        --runs qwen38_gold_think_s0 qwen38_gold_think_s1 ... --out logs/groundedness/claims.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.groundedness import extract_claims  # noqa: E402

DEFAULT_RUNS = (
    [f"qwen38_gold_think_s{s}" for s in range(3)]
    + [f"qwen38_gold_ctx32k_s{s}" for s in range(3)]
    + [f"qwen38_gold_ctx128k_s{s}" for s in range(3)]
    + [f"qwen_baseline_gold_s{s}" for s in range(3)]
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="logs_cineca/qwen38")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="file stems (without .jsonl) in --records-dir")
    ap.add_argument("--out", default="logs/groundedness/claims.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_claims = n_skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for run in args.runs:
            path = os.path.join(args.records_dir, run + ".jsonl")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    answer = next((m["content"] for m in record["messages"] if m["role"] == "assistant"), "")
                    try:
                        parsed = json.loads(answer)
                    except json.JSONDecodeError:
                        parsed = None
                    if not isinstance(parsed, dict):
                        n_skipped += 1
                        continue
                    paper = os.path.basename(record["source_file"])
                    for k, c in enumerate(extract_claims(parsed)):
                        out.write(json.dumps({"id": f"{run}|{paper}|{k}", "run": run, "provider": record.get("provider"),
                                              "source_file": paper, **c}, ensure_ascii=False) + "\n")
                        n_claims += 1
    print(f"[+] {n_claims} claims from {len(args.runs)} runs ({n_skipped} unparsable answers skipped) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
