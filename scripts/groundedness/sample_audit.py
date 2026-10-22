"""
Draws the claims to audit by hand: a fixed number per provider x judge band
(stratified, seeded), with the 3 paper sentences closest to each claim. Writes
audit_sample.json (all fields, for the annotator tooling) and audit_sample.md
(no judge band shown, blank label to fill: TRUE / PARTIAL / FALSE).

    python scripts/groundedness/sample_audit.py --claim-status logs/groundedness_atomic/claim_status.jsonl \
        --records-dir logs_cineca/qwen38 --per-cell 14 --providers think baseline --out-dir logs/groundedness_atomic
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.judges import BM25, load_papers, retrieve_passages, split_sentences  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim-status", required=True)
    ap.add_argument("--records-dir", default="logs_cineca/qwen38")
    ap.add_argument("--per-cell", type=int, default=14, help="claims per provider x band")
    ap.add_argument("--providers", nargs="+", help="substrings of the provider names to keep (default: all)")
    ap.add_argument("--exclude", nargs="*", default=[], help="JSONL files whose claim ids are not drawn again")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.claim_status, encoding="utf-8") as f:
        claims = [json.loads(line) for line in f if line.strip()]
    skip = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as f:
            skip |= {json.loads(line)["id"] for line in f if line.strip()}
    if args.providers:
        claims = [c for c in claims if any(p in c["provider"] for p in args.providers)]
    cells = defaultdict(list)
    for c in claims:
        if c["id"] not in skip:
            cells[(c["provider"], c["status"])].append(c)

    rng = random.Random(args.seed)
    sample = []
    for key in sorted(cells):
        pool = cells[key]
        sample += rng.sample(pool, min(args.per_cell, len(pool)))
        print(f"  {key[0]:<38} {key[1]:<16} {len(pool):>5} claims, {min(args.per_cell, len(pool))} drawn")
    rng.shuffle(sample)   # the annotator must not see the bands in order

    papers, index = {}, {}
    out = []
    for n, c in enumerate(sample):
        run = c["id"].split("|")[0]
        if run not in papers:
            papers[run] = load_papers(os.path.join(args.records_dir, run + ".jsonl"))
        if (run, c["source_file"]) not in index:
            sents = split_sentences(papers[run][c["source_file"]])
            index[(run, c["source_file"])] = (sents, BM25(sents))
        sents, bm = index[(run, c["source_file"])]
        out.append({"n": n, **{k: c[k] for k in ("id", "provider", "source_file", "field", "claim", "status", "p")},
                    "passages": retrieve_passages(sents, bm, c["claim"], top_k=4, window=1)})

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "audit_sample.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "audit_sample.md"), "w", encoding="utf-8") as f:
        f.write("# Audit sample\n\nLabel each claim against the paper: TRUE (fully supported), PARTIAL (a detail is not in the "
                "paper, wrongly attributed or overstated), FALSE (contradicted or inverted).\n")
        for x in out:
            f.write(f"\n## #{x['n']} [{x['field']}] {x['source_file'][:30]}\n**Claim:** {x['claim']}\n\n**Label:** ____\n")
            for p in x["passages"][:3]:
                f.write(f"- paper: {p[:400]}\n")
    print(f"[+] {len(out)} claims -> {args.out_dir}/audit_sample.json and audit_sample.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
