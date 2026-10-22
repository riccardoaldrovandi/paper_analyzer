"""
Checks gold/claims.jsonl against the paper text the model saw.

SUPPORTED / CONTRADICTED claims must carry an "evidence" quote that appears
verbatim in the paper (after NFKC, whitespace and quote normalization).
NOT_ENOUGH_INFO claims may list "absent_terms" that must NOT appear in it.

Usage: python scripts/gold/check_claims.py gold/claims.jsonl <export-dataset .jsonl>
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.metrics import split_source  # noqa: E402

LABELS = {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("−", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text)


def main(claims_path: str, records_path: str) -> int:
    papers = {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                user = next(m["content"] for m in r["messages"] if m["role"] == "user")
                papers[os.path.basename(r["source_file"])] = norm(split_source(user)[0])
    problems, counts = 0, Counter()
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            counts[c["label"]] += 1
            text = papers.get(c["source_file"])
            if text is None:
                print(f"{c['id']}: no record for {c['source_file']}")
                problems += 1
            elif c["label"] not in LABELS:
                print(f"{c['id']}: unknown label {c['label']!r}")
                problems += 1
            elif c["label"] != "NOT_ENOUGH_INFO":
                if norm(c["evidence"]) not in text:
                    print(f"{c['id']}: evidence quote not found in the paper")
                    problems += 1
            else:
                for term in c.get("absent_terms", []):
                    if term.lower() in text.lower():
                        print(f"{c['id']}: term {term!r} is in the paper")
                        problems += 1
    print(f"{sum(counts.values())} claims {dict(counts)}, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
