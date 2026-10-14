import json
from collections import defaultdict

from src.evaluation.metrics import evaluate_record, normalize_answer


def run_evaluation(input_path: str, report_path: str, filtered_path: str = None, min_l2: float = 0.8) -> None:
    """
    Scores every record of an export-dataset JSONL file, writes one line of
    metrics per record to report_path and prints a per-provider summary. With
    filtered_path, also writes the records that pass Level 1 and score at least
    min_l2 on Level 2 (answers normalized to one schema), ready for fine-tuning.
    The input file is never modified.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[+] Evaluating {len(records)} record(s) from '{input_path}'.")

    by_provider = defaultdict(lambda: {"total": 0, "l1_pass": 0, "l2_scores": [], "kept": 0,
                                       "figures": 0, "formulas_available": 0, "formulas_reported": 0, "empty_fields": 0})
    kept = []
    with open(report_path, "w", encoding="utf-8") as rep_f:
        for record in records:
            result = evaluate_record(record)
            l1, l2 = result["level1"], result["level2"]
            # A record with nothing checkable at Level 2 (score None) can't be
            # shown wrong, so it is kept; the report still says so.
            keep = l1["passed"] and (l2.get("score") is None or l2["score"] >= min_l2)
            result["kept"] = keep
            rep_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            stats = by_provider[result["provider"]]
            stats["total"] += 1
            stats["l1_pass"] += l1["passed"]
            if l2.get("score") is not None:
                stats["l2_scores"].append(l2["score"])
            if keep:
                stats["kept"] += 1
                kept.append(record)
            cov = result["coverage"]
            if cov:
                stats["figures"] += cov["figures_with_data"]
                stats["formulas_available"] += cov["formulas_available"]
                stats["formulas_reported"] += cov["formulas_reported"]
                stats["empty_fields"] += len(cov["empty_fields"])

            if not keep:
                why = l1["reasons"] or [f"L2 score {l2['score']}"]
                print(f"[!] Discarded {result['source_file']} ({result['provider']}): {'; '.join(why)}")

    print(f"\n{'provider':<40} {'records':>7} {'L1 pass':>8} {'L2 mean':>8} {'kept':>5} {'figures':>8} {'formulas':>9} {'empty':>6}")
    for provider, s in sorted(by_provider.items(), key=lambda x: -x[1]["total"]):
        mean = f"{sum(s['l2_scores']) / len(s['l2_scores']):.3f}" if s["l2_scores"] else "n/a"
        formulas = f"{s['formulas_reported']}/{s['formulas_available']}"
        print(f"{str(provider):<40} {s['total']:>7} {s['l1_pass']:>8} {mean:>8} {s['kept']:>5} {s['figures']:>8} {formulas:>9} {s['empty_fields']:>6}")
    print("figures = figures with usable data passed to the model; formulas = PySR formulas the model reported / available; "
          "empty = paper fields left empty (total)")
    print(f"\n[+] Per-record metrics written to '{report_path}'.")

    if filtered_path:
        with open(filtered_path, "w", encoding="utf-8") as out_f:
            for record in kept:
                record = json.loads(json.dumps(record))  # don't touch the caller's copy
                for m in record["messages"]:
                    if m["role"] == "assistant":
                        m["content"] = normalize_answer(m["content"])
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[+] {len(kept)}/{len(records)} record(s) kept (L1 passed, L2 >= {min_l2}) -> '{filtered_path}'.")
