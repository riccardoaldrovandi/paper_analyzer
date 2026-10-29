import json
from collections import defaultdict

from src.evaluation.metrics import evaluate_record, normalize_answer, split_source
from src.prompt_v2 import is_v2_record, record_version
from src.evaluation.rubric import load_rubrics, rubric_for, check_rubric
from src.evaluation.figure_checks import load_figure_checks, checks_for, score_figure_checks
from src.evaluation.stats import print_statistics, N_BOOTSTRAP


def run_evaluation(input_paths, report_path: str, filtered_path: str = None, min_l2: float = 0.8,
                   rubrics_path: str = None, n_bootstrap: int = N_BOOTSTRAP,
                   figure_checks_path: str = None) -> None:
    """
    Scores every record of one or more export-dataset JSONL files, writes one line of
    metrics per record to report_path and prints a per-provider summary. With
    filtered_path, also writes the records that pass Level 1 and score at least
    min_l2 on Level 2 (answers normalized to one schema), ready for fine-tuning.
    The input files are never modified.

    With rubrics_path, records of papers that have a rubric also get the
    rubric recall (completeness) and are left out of the filtered output:
    those papers are the held-out gold set and must not end up in training.

    With figure_checks_path, v2 answers of papers with figure checks also get
    the share of figure-reading checks they pass (see figure_checks.py).

    Several input files are scored together: records of the same provider on
    the same paper count as repeated runs (see stats.py), and every pair of
    providers gets a paired comparison with a bootstrap confidence interval.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    records = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"[+] Evaluating {len(records)} record(s) from {', '.join(repr(p) for p in input_paths)}.")
    rubrics = load_rubrics(rubrics_path) if rubrics_path else {}
    if rubrics_path:
        print(f"[+] Loaded {len(rubrics)} rubric(s) from '{rubrics_path}'.")
    checked = set()
    fig_checks = load_figure_checks(figure_checks_path) if figure_checks_path else {}
    if figure_checks_path:
        print(f"[+] Loaded {sum(len(v) for v in fig_checks.values())} figure check(s) on {len(fig_checks)} paper(s) from '{figure_checks_path}'.")
    fig_stats = defaultdict(lambda: {"records": 0, "checks": 0, "passed": 0, "visual_checks": 0, "visual_passed": 0, "wrong": 0, "not_described": 0,
                                     "verified": 0, "passed_before": 0, "checks_before": 0})

    by_provider = defaultdict(lambda: {"total": 0, "l1_pass": 0, "l2_scores": [], "kept": 0,
                                       "figures": 0, "formulas_available": 0, "formulas_reported": 0, "empty_fields": 0,
                                       "rubric_recalls": [], "rubric_strict": 0})
    kept, per_record = [], []
    with open(report_path, "w", encoding="utf-8") as rep_f:
        for record in records:
            rubric = rubric_for(rubrics, record.get("source_file"))
            if rubric and rubric["source_file"] not in checked:
                checked.add(rubric["source_file"])
                user_content = next((m["content"] for m in record.get("messages", []) if m.get("role") == "user"), "")
                for warning in check_rubric(rubric, split_source(user_content)[0]):
                    print(f"[!] Rubric {rubric['source_file']}: {warning}")
            result = evaluate_record(record, rubric)
            checks = checks_for(fig_checks, record.get("source_file")) if is_v2_record(record) else []
            if checks:
                user_content = next((m["content"] for m in record.get("messages", []) if m.get("role") == "user"), "")
                answer = next((m["content"] for m in record.get("messages", []) if m.get("role") == "assistant"), "")
                try:
                    parsed = json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                fc = result["figure_checks"] = score_figure_checks(parsed, checks, split_source(user_content)[0])
                fs = fig_stats[result["provider"]]
                fs["records"] += 1
                for k in ("checks", "passed", "visual_checks", "visual_passed"):
                    fs[k] += fc[k]
                fs["wrong"] += len(fc["wrong"])
                fs["not_described"] += len(fc["not_described"])
                if record.get("figures_before_check") is not None:
                    # VERIFY_FIGURES=1: the same answer before the verification pass.
                    before = score_figure_checks(dict(parsed, figures=record["figures_before_check"]), checks, split_source(user_content)[0])
                    result["figure_checks_before_verification"] = before
                    fs["verified"] += 1
                    fs["passed_before"] += before["passed"]
                    fs["checks_before"] += before["checks"]
            l1, l2 = result["level1"], result["level2"]
            # A record with nothing checkable at Level 2 (score None) can't be
            # shown wrong, so it is kept; the report still says so.
            keep = l1["passed"] and (l2.get("score") is None or l2["score"] >= min_l2)
            held_out = rubric is not None
            result["kept"] = keep and not held_out
            rep_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            per_record.append({"provider": result["provider"], "source_file": result["source_file"],
                               "l2": l2.get("score"), "recall": result.get("rubric", {}).get("recall")})
            stats = by_provider[result["provider"]]
            stats["total"] += 1
            stats["l1_pass"] += l1["passed"]
            if l2.get("score") is not None:
                stats["l2_scores"].append(l2["score"])
            if keep and not held_out:
                stats["kept"] += 1
                kept.append(record)
            if result.get("rubric", {}).get("recall") is not None:
                stats["rubric_recalls"].append(result["rubric"]["recall"])
                stats["rubric_strict"] += result["rubric"]["strict"]
            cov = result["coverage"]
            if cov:
                stats["figures"] += cov["figures_with_data"]
                stats["formulas_available"] += cov["formulas_available"]
                stats["formulas_reported"] += cov["formulas_reported"]
                stats["empty_fields"] += len(cov["empty_fields"])

            if not keep:
                why = l1["reasons"] or [f"L2 score {l2['score']}"]
                print(f"[!] Discarded {result['source_file']} ({result['provider']}): {'; '.join(why)}")

    rubric_cols = f" {'recall':>7} {'strict':>7}" if rubrics else ""
    print(f"\n{'provider':<40} {'records':>7} {'L1 pass':>8} {'L2 mean':>8} {'kept':>5} {'figures':>8} {'formulas':>9} {'empty':>6}{rubric_cols}")
    for provider, s in sorted(by_provider.items(), key=lambda x: -x[1]["total"]):
        mean = f"{sum(s['l2_scores']) / len(s['l2_scores']):.3f}" if s["l2_scores"] else "n/a"
        formulas = f"{s['formulas_reported']}/{s['formulas_available']}"
        row = ""
        if rubrics:
            r = s["rubric_recalls"]
            recall = f"{sum(r) / len(r):.3f}" if r else "n/a"
            strict = f"{s['rubric_strict']}/{len(r)}" if r else "n/a"
            row = f" {recall:>7} {strict:>7}"
        print(f"{str(provider):<40} {s['total']:>7} {s['l1_pass']:>8} {mean:>8} {s['kept']:>5} {s['figures']:>8} {formulas:>9} {s['empty_fields']:>6}{row}")
    print("figures = figures with usable data passed to the model; formulas = PySR formulas the model reported / available; "
          "empty = paper fields left empty (total)")
    if rubrics:
        print("recall = weighted share of rubric facts found (mean over papers with a rubric); "
              "strict = papers with every fact found. Papers with a rubric are held out of the filtered output.")
    if fig_stats:
        print(f"\n{'provider':<40} {'records':>7} {'fig checks':>11} {'visual only':>12} {'wrong':>6} {'fig. not described':>19}")
        for provider, s in sorted(fig_stats.items()):
            share = lambda a, b: f"{a}/{b} {a / b:.0%}" if b else "n/a"
            print(f"{str(provider):<40} {s['records']:>7} {share(s['passed'], s['checks']):>11} "
                  f"{share(s['visual_passed'], s['visual_checks']):>12} {s['wrong']:>6} {s['not_described']:>19}")
        for provider, s in sorted(fig_stats.items()):
            if s["verified"]:
                print(f"  {provider}: before the figure verification pass {s['passed_before']}/{s['checks_before']} "
                      f"({s['passed_before'] / s['checks_before']:.0%}) on {s['verified']} verified record(s)")
        print("fig checks = figure-reading checks passed; visual only = the checks the paper's text can't answer; "
              "wrong = checks failed by a contradicting statement; fig. not described = figures with checks the answer omits")
    print_statistics(per_record, n_bootstrap)
    print(f"\n[+] Per-record metrics written to '{report_path}'.")

    if filtered_path:
        with open(filtered_path, "w", encoding="utf-8") as out_f:
            for record in kept:
                record = json.loads(json.dumps(record))  # don't touch the caller's copy
                version = record_version(record)
                for m in record["messages"]:
                    if m["role"] == "assistant":
                        m["content"] = normalize_answer(m["content"], version)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        held = " and not held out by a rubric" if rubrics else ""
        print(f"[+] {len(kept)}/{len(records)} record(s) kept (L1 passed, L2 >= {min_l2}{held}) -> '{filtered_path}'.")
