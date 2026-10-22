"""
Runs ONE groundedness judge on every claim of gold/claims.jsonl and writes its
predictions. Needs the GPU environment (see scripts/nli/setup_env.sh); scoring
the predictions needs no GPU (scripts/nli/score_judges.py).

    python scripts/nli/benchmark_judges.py --judge minicheck-flan-t5 \
        --claims gold/claims.jsonl --records logs_cineca/qwen38/qwen38_gold_ctx128k_s0.jsonl \
        --out-dir logs/nli_predictions
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.judges import JUDGES, load_papers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--claims", default="gold/claims.jsonl")
    ap.add_argument("--records", required=True, help="export-dataset JSONL holding the paper texts the model saw")
    ap.add_argument("--out-dir", default="logs/nli_predictions")
    ap.add_argument("--cache-dir", default=os.environ.get("NLI_HF_CACHE"), help="Hugging Face cache with the models")
    args = ap.parse_args()

    papers = load_papers(args.records)
    with open(args.claims, encoding="utf-8") as f:
        claims = [json.loads(line) for line in f if line.strip()]
    missing = sorted({c["source_file"] for c in claims} - set(papers))
    if missing:
        print(f"[!] No paper text in {args.records} for: {missing}")
        return 1

    print(f"[~] Loading judge {args.judge} (cache: {args.cache_dir})")
    t0 = time.time()
    judge = JUDGES[args.judge](args.cache_dir)
    print(f"[+] Loaded in {time.time() - t0:.0f}s; judging {len(claims)} claims")

    t0 = time.time()
    results = judge.judge([papers[c["source_file"]] for c in claims], [c["claim"] for c in claims])
    elapsed = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.judge}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for claim, res in zip(claims, results):
            f.write(json.dumps({"id": claim["id"], "judge": args.judge, **res}, ensure_ascii=False) + "\n")
    print(f"[+] {len(results)} predictions in {elapsed:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
