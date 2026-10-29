"""The fine-tuning files from the reviewed queue (ft_review/queue.jsonl).

Only papers marked "accepted" or "edited" in review_app.py are used; for
"edited" the corrected answer replaces the teacher's. One chat example per
paper (system = the v2.1 prompt the teacher saw, user = the paper text it saw,
assistant = the reviewed JSON), plus the image file names when the answer came
from a multimodal mode (the images themselves stay in the generation
checkpoint dirs on Leonardo). A validation split is drawn per arXiv category
(--val-fraction, fixed --seed) so both splits cover every subfield.

    python scripts/finetune/build_training_set.py [--val-fraction 0.1]
    -> ft_review/train.jsonl, ft_review/val.jsonl, ft_review/stats.json
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join("ft_review", "queue.jsonl"))
    ap.add_argument("--out-dir", default="ft_review")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.queue, encoding="utf-8") if line.strip()]
    status = Counter(i["status"] for i in items)
    usable = [i for i in items if i["status"] in ("accepted", "edited")]
    by_cat = defaultdict(list)
    for i in usable:
        by_cat[i.get("category") or "unknown"].append(i)

    rng = random.Random(args.seed)
    train, val = [], []
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda i: i["paper"])
        rng.shuffle(group)
        n_val = round(len(group) * args.val_fraction)
        val += group[:n_val]
        train += group[n_val:]

    def example(i):
        answer = i["reviewed_answer"] if i["status"] == "edited" else i["answer"]
        return {
            "paper": i["paper"], "arxiv_id": i.get("arxiv_id"), "category": i.get("category"),
            "review": i["status"], "images": i.get("images", []),
            "messages": [
                {"role": "system", "content": i["system"]},
                {"role": "user", "content": i["user"]},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
        }

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for i in split:
                f.write(json.dumps(example(i), ensure_ascii=False) + "\n")
    stats = {"queue": dict(status), "train": len(train), "val": len(val),
             "edited_share": round(status["edited"] / max(len(usable), 1), 3),
             "per_category": {c: len(v) for c, v in sorted(by_cat.items())},
             "rejection_notes": [i["note"] for i in items if i["status"] == "rejected"]}
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[+] review status {dict(status)} -> {len(train)} train, {len(val)} val example(s) in {args.out_dir}/")
    if status["pending"]:
        print(f"[~] {status['pending']} paper(s) still to review.")


if __name__ == "__main__":
    main()
