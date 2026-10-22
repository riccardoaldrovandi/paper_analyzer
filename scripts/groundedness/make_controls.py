"""
Step (CPU, PC): builds the control claims of src/evaluation/controls.py from the
judged atomic claims. Output has the same fields as claims.jsonl, so
judge_claims.py scores it unchanged (for a negative control the claim carries the
OTHER paper in `source_file`, which is the one the judge reads).

    python scripts/groundedness/make_controls.py --claims logs/groundedness/atomic_claims.jsonl \
        --scores logs/groundedness/atomic_scores.jsonl --records-dir logs_cineca/qwen38 \
        --out logs/groundedness/controls_claims.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.controls import flip_word, group_of, negate, swap_number  # noqa: E402
from src.evaluation.groundedness import VERIFIED_MIN  # noqa: E402
from src.evaluation.judges import load_papers  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-negative", type=int, default=400)
    ap.add_argument("--n-perturbed", type=int, default=150, help="per kind")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    claims = {c["id"]: c for c in read_jsonl(args.claims)}
    p = {s["id"]: s["p_supported"] for s in read_jsonl(args.scores)}
    rng = random.Random(args.seed)
    runs = sorted({c["run"] for c in claims.values()})
    papers = {r: load_papers(os.path.join(args.records_dir, r + ".jsonl")) for r in runs}
    ids = sorted(i for i in claims if i in p)
    rng.shuffle(ids)

    out = []
    # negative control: any claim, judged against a paper of another topic group
    for i in ids[:args.n_negative]:
        c = claims[i]
        others = sorted(f for f in papers[c["run"]] if group_of(f) != group_of(c["source_file"]))
        other = rng.choice(others)
        out.append({**c, "id": f"neg|{i}", "kind": "negative", "orig_id": i, "source_file": other,
                    "orig_source_file": c["source_file"], "orig_p": p[i]})
    # perturbations of claims the judge called VERIFIED
    verified = [i for i in ids if p[i] >= VERIFIED_MIN]
    for kind, fn in (("number", lambda c, t: swap_number(c, t)), ("flip", lambda c, t: flip_word(c)),
                     ("negate", lambda c, t: negate(c))):
        n = 0
        for i in verified:
            c = claims[i]
            new = fn(c["claim"], papers[c["run"]][c["source_file"]])
            if new and new != c["claim"]:
                out.append({**c, "id": f"{kind}|{i}", "kind": kind, "orig_id": i, "claim": new,
                            "orig_claim": c["claim"], "orig_p": p[i]})
                n += 1
            if n >= args.n_perturbed:
                break
        print(f"[~] {kind}: {n} perturbed claims (of {len(verified)} VERIFIED)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[+] {len(out)} control claims -> {args.out}")


if __name__ == "__main__":
    main()
