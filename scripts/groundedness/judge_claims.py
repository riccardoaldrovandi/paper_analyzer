"""
Step 2 (GPU, Leonardo): gives each claim of claims.jsonl a probability that its
paper supports it. The paper text of each run comes from <records-dir>/<run>.jsonl.
Resumable: claims already in --out are skipped, so a job that hits its time
limit loses nothing.

    python scripts/groundedness/judge_claims.py --claims logs/groundedness/claims.jsonl \
        --records-dir logs_cineca/qwen38 --out logs/groundedness/scores.jsonl
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.judges import JUDGES, load_papers  # noqa: E402

CHUNK = 64   # claims per write: bounds what a crash can lose


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="bespoke-minicheck-7b", choices=sorted(JUDGES))
    ap.add_argument("--cache-dir", default=os.environ.get("NLI_HF_CACHE"))
    args = ap.parse_args()

    with open(args.claims, encoding="utf-8") as f:
        claims = [json.loads(line) for line in f if line.strip()]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [c for c in claims if c["id"] not in done]
    print(f"[~] {len(claims)} claims, {len(done)} already scored, {len(todo)} to do")
    if not todo:
        return 0

    by_run = defaultdict(list)
    for c in todo:
        by_run[c["run"]].append(c)

    print(f"[~] Loading judge {args.judge} (cache: {args.cache_dir})")
    t0 = time.time()
    judge = JUDGES[args.judge](args.cache_dir)
    print(f"[+] Loaded in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scored = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for run, run_claims in sorted(by_run.items()):
            papers = load_papers(os.path.join(args.records_dir, run + ".jsonl"))
            run_claims.sort(key=lambda c: c["source_file"])   # same paper consecutively
            for i in range(0, len(run_claims), CHUNK):
                chunk = run_claims[i:i + CHUNK]
                results = judge.judge([papers[c["source_file"]] for c in chunk], [c["claim"] for c in chunk])
                for c, r in zip(chunk, results):
                    out.write(json.dumps({"id": c["id"], "p_supported": r["scores"]["supported"], "label": r["label"]}) + "\n")
                out.flush()
                scored += len(chunk)
                print(f"[~] {run}: {scored}/{len(todo)} claims, {time.time() - t0:.0f}s", flush=True)
    print(f"[+] Done: {scored} claims in {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
