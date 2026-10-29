"""Picks, for every corpus paper, the best teacher answer among the seeds: the review queue.

Input: the export-dataset JSONL files of scripts/finetune/generate.sbatch
(several seeds of the same papers). For each paper:
  1. answers failing Level 1 (invalid / incomplete JSON, schema echo, filler,
     repetition) are dropped;
  2. among the rest, the highest Level 2 without the visual sub-score wins
     (numbers, references and datasets grounded in the paper; v2.1 fills the
     PySR field itself), ties broken by fewer empty fields;
  3. answers below --min-l2 are kept out of the queue (listed in the summary):
     reviewing a badly grounded answer costs more than it gives.
The cross-seed agreement on the numbers cited (Jaccard) is stored for the
reviewer: a low value marks papers where the teacher was unsure.

Output: ft_review/queue.jsonl, one line per paper, read by review_app.py.
An existing queue keeps its review state (status, edits): only new papers are
added, so select_answers.py can be rerun when more records arrive.

    python scripts/finetune/select_answers.py --input logs_cineca/ft_data/*.jsonl [--min-l2 0.8]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.evaluation.metrics import evaluate_record, _numbers, _as_text  # noqa: E402

QUEUE = os.path.join("ft_review", "queue.jsonl")


def l2_without_visual(l2: dict):
    scores = [l2[k]["score"] for k in ("numbers", "references", "datasets") if l2.get(k, {}).get("score") is not None]
    return sum(scores) / len(scores) if scores else None


def local_path(source_file: str, corpus: str) -> str:
    """Leonardo path (corpus_ft/shards_pdf_6/3/<name>) -> the same file in the local corpus."""
    name = os.path.basename(source_file)
    sub = "latex" if name.endswith(".tar.gz") else "pdfs"
    return os.path.join(corpus, sub, name)


def paper_key(source_file: str) -> str:
    return re.sub(r"(_source)?\.(pdf|tar\.gz)$", "", os.path.basename(source_file))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--min-l2", type=float, default=0.8)
    ap.add_argument("--corpus", default="corpus_ft")
    ap.add_argument("--queue", default=QUEUE)
    args = ap.parse_args()

    by_paper = defaultdict(list)
    for path in args.input:
        if path.endswith(".rejected.jsonl"):
            continue
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                by_paper[paper_key(rec["source_file"])].append(rec)

    queue = {}
    if os.path.exists(args.queue):
        for line in open(args.queue, encoding="utf-8"):
            item = json.loads(line)
            queue[item["paper"]] = item
    manifest = {}
    mpath = os.path.join(args.corpus, "manifest.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath, encoding="utf-8"):
            m = json.loads(line)
            if m.get("pdf"):
                manifest[paper_key(m["pdf"])] = m

    added, low, failed = 0, [], []
    for paper, records in sorted(by_paper.items()):
        if paper in queue:
            continue
        scored = []
        for rec in records:
            res = evaluate_record(rec)
            if not res["level1"]["passed"]:
                continue
            answer = json.loads(rec["messages"][-1]["content"])
            empty = len((res.get("coverage") or {}).get("empty_fields", []))
            scored.append((l2_without_visual(res["level2"]) or 0.0, -empty, rec, res, answer))
        if not scored:
            failed.append(paper)
            continue
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        l2, _, rec, res, answer = scored[0]
        nums = [_numbers(_as_text({k: v for k, v in a.items() if k != "visual_analysis_findings"})) for *_, a in scored]
        agreement = None
        if len(nums) > 1:
            pairs = [(a, b) for i, a in enumerate(nums) for b in nums[i + 1:]]
            agreement = round(sum(len(a & b) / max(len(a | b), 1) for a, b in pairs) / len(pairs), 3)
        if l2 < args.min_l2:
            low.append((paper, round(l2, 3)))
            continue
        meta = manifest.get(paper, {})
        queue[paper] = {
            "paper": paper,
            "title": meta.get("title"), "arxiv_id": meta.get("arxiv_id"), "category": meta.get("category"),
            "source_file": rec["source_file"],
            "local_pdf": meta.get("pdf") or local_path(rec["source_file"], args.corpus),
            "seed": (rec.get("generation") or {}).get("seed"),
            "candidates": len(records), "valid": len(scored),
            "metrics": {"l2_no_visual": round(l2, 3), "level2": res["level2"], "empty_fields": res["coverage"]["empty_fields"] if res.get("coverage") else [],
                        "seed_number_agreement": agreement},
            "system": rec["messages"][0]["content"],
            "user": rec["messages"][1]["content"],
            "images": rec.get("images", []),
            "answer": answer,
            "status": "pending", "reviewed_answer": None, "note": "",
        }
        added += 1

    os.makedirs(os.path.dirname(args.queue) or ".", exist_ok=True)
    with open(args.queue, "w", encoding="utf-8") as f:
        for item in queue.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[+] {len(by_paper)} paper(s) in the input, {added} added to the queue ({len(queue)} total) -> {args.queue}")
    if low:
        print(f"[-] {len(low)} paper(s) left out, best L2 (no visual) below {args.min_l2}: {low}")
    if failed:
        print(f"[-] {len(failed)} paper(s) with no answer passing Level 1: {failed}")


if __name__ == "__main__":
    main()
