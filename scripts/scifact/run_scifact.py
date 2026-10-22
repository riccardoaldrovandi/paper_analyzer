"""
The groundedness judge on SciFact (Wadden et al., 2020), the benchmark whose
scheme this system adapts (docs/METRICS.md 7.4). GPU script (Leonardo).

Setting: ORACLE abstracts. For each dev claim the premise is the abstract of the
document(s) the claim was written about (`cited_doc_ids`), so the judge is tested
on verification only, not on retrieval.
    gold SUPPORTED     the claim has SUPPORT evidence
    gold CONTRADICTED  the claim has CONTRADICT evidence
    gold NOT_ENOUGH_INFO  no evidence
The judge is binary (supported vs not); metrics as in calibration.py, plus the
share of CONTRADICTED and NEI claims it correctly refuses to call supported.

    python scripts/scifact/run_scifact.py --data $WORK/scifact/data --out logs/scifact_bespoke.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.calibration import evaluate  # noqa: E402
from src.evaluation.judges import JUDGES  # noqa: E402


def load(data_dir):
    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["doc_id"])] = d["title"] + ". " + " ".join(d["abstract"])
    items = []
    with open(os.path.join(data_dir, "claims_dev.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            labels = {e["label"] for ev in c.get("evidence", {}).values() for e in ev}
            label = "SUPPORTED" if "SUPPORT" in labels else "CONTRADICTED" if "CONTRADICT" in labels else "NOT_ENOUGH_INFO"
            docs = [str(d) for d in (c.get("cited_doc_ids") or list(c.get("evidence", {})))]
            docs = [d for d in docs if d in corpus]
            if not docs:
                continue
            items.append({"id": str(c["id"]), "claim": c["claim"], "label": label, "origin": label,
                          "doc": "\n\n".join(corpus[d] for d in docs)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="bespoke-minicheck-7b", choices=sorted(JUDGES))
    ap.add_argument("--cache-dir", default=os.environ.get("NLI_HF_CACHE"))
    args = ap.parse_args()
    items = load(args.data)
    print(f"[~] {len(items)} SciFact dev claims: " + ", ".join(f"{l} {sum(i['label'] == l for i in items)}"
          for l in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")))
    judge = JUDGES[args.judge](args.cache_dir)
    t0 = time.time()
    res = judge.judge([i["doc"] for i in items], [i["claim"] for i in items])
    print(f"[+] judged in {time.time() - t0:.0f}s")
    preds = {i["id"]: r["label"] for i, r in zip(items, res)}
    summary = evaluate(items, preds)
    summary["judge"] = args.judge
    summary["setting"] = "oracle abstracts (cited_doc_ids)"
    summary["p_supported"] = {i["id"]: r["scores"].get("supported") for i, r in zip(items, res)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    b = summary["binary"]
    print(f"[+] binary accuracy {b['accuracy'][0]:.3f} [{b['accuracy'][1]:.3f}, {b['accuracy'][2]:.3f}], "
          f"balanced {b['balanced_accuracy'][0]:.3f}, macro-F1 {b['macro_f1'][0]:.3f}, kappa {b['kappa'][0]:.3f}")
    print("    accuracy by gold class:", {k: round(v["binary_accuracy"], 3) for k, v in summary["binary_by_origin"].items()})


if __name__ == "__main__":
    main()
