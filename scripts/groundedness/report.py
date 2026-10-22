"""
Step 3 (CPU, run on the PC): scorecard of the extractions, combining
- groundedness (judge scores + number check + doubt band, src/evaluation/groundedness.py),
- the existing deterministic metrics (L1, L2) and the rubric recall (completeness),
  read from an `evaluate` report (e.g. logs_cineca/qwen38/eval_report.jsonl).

    python scripts/groundedness/report.py --claims logs/groundedness/claims.jsonl \
        --scores logs/groundedness/scores.jsonl --records-dir logs_cineca/qwen38 \
        --eval-report logs_cineca/qwen38/eval_report.jsonl --audit gold/audit_qwen38.jsonl --out-dir logs/groundedness
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.groundedness import (CLAIM_FIELDS, STATUSES, status, stratified_difference,  # noqa: E402
                                         stratified_estimate, summarize, ungrounded_numbers)
from src.evaluation.judges import BM25, load_papers, retrieve_passages, split_sentences  # noqa: E402
from src.evaluation.stats import N_BOOTSTRAP, bootstrap_ci  # noqa: E402


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_ci(values: list) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) < 2:
        return f"{values[0]:.3f} (1 paper)"
    m, lo, hi = bootstrap_ci(values, N_BOOTSTRAP)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def per_paper_mean(rows: list, key) -> dict:
    """{paper: mean of key(row) over the runs of that paper}: a paper never weighs more for having more runs."""
    by_paper = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is not None:
            by_paper[r["source_file"]].append(v)
    return {p: statistics.mean(v) for p, v in by_paper.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--records-dir", default="logs_cineca/qwen38")
    ap.add_argument("--eval-report", help="evaluate report with L1/L2 and rubric recall per record")
    ap.add_argument("--audit", help="audited sample of claims (gold/audit_qwen38.jsonl) for the error-rate estimate")
    ap.add_argument("--out-dir", default="logs/groundedness")
    args = ap.parse_args()

    claims = read_jsonl(args.claims)
    scores = {r["id"]: r["p_supported"] for r in read_jsonl(args.scores)}
    missing = [c["id"] for c in claims if c["id"] not in scores]
    if missing:
        print(f"[!] {len(missing)}/{len(claims)} claims have no score (job not finished?): they are left out.")
    claims = [c for c in claims if c["id"] in scores]

    papers, indexes = {}, {}
    for run in sorted({c["run"] for c in claims}):
        papers[run] = load_papers(os.path.join(args.records_dir, run + ".jsonl"))
    for c in claims:
        text = papers[c["run"]][c["source_file"]]
        c["p"] = scores[c["id"]]
        c["ungrounded"] = ungrounded_numbers(c["claim"], text)
        c["status"] = status(c["p"], c["ungrounded"])

    # One row per answer (run x paper).
    answers = defaultdict(list)
    for c in claims:
        answers[(c["provider"], c["run"], c["source_file"])].append(c)
    rows = []
    for (provider, run, paper), cs in answers.items():
        st = [c["status"] for c in cs]
        s = summarize(st)
        rows.append({"provider": provider, "run": run, "source_file": paper, "claims": s["claims"],
                     "verified": s["verified"], "doubtful": s["doubtful"], "number_mismatch": s["number_mismatch"],
                     "unsupported": s["unsupported"], "flagged": s["flagged"]})
    providers = sorted({r["provider"] for r in rows}, key=str)

    # Existing metrics per (provider, paper).
    ev = defaultdict(list)
    if args.eval_report:
        for r in read_jsonl(args.eval_report):
            ev[r["provider"]].append({"source_file": os.path.basename(r["source_file"]),
                                      "l1": float(r["level1"]["passed"]), "l2": r["level2"].get("score"),
                                      "recall": (r.get("rubric") or {}).get("recall")})

    summary = {}
    print(f"Scorecard, mean over papers [95% bootstrap CI over papers]; {len(claims)} claims scored\n")
    print("Judge signal (NOT error rates, see the audit below) next to the deterministic metrics and the completeness:")
    cols = ("VERIFIED", "DOUBTFUL", "flagged", "L2", "rubric recall")
    print(f"{'provider':<38} " + " ".join(f"{c:>22}" for c in cols) + f" {'claims/answer':>14}")
    for prov in providers:
        pr = [r for r in rows if r["provider"] == prov]
        ver = per_paper_mean(pr, lambda r: r["verified"])
        dbt = per_paper_mean(pr, lambda r: r["doubtful"])
        flg = per_paper_mean(pr, lambda r: r["flagged"])
        l2 = per_paper_mean(ev[prov], lambda r: r["l2"])
        rec = per_paper_mean(ev[prov], lambda r: r["recall"])
        summary[prov] = {"answers": len(pr), "papers": len(ver), "claims_per_answer": statistics.mean(r["claims"] for r in pr),
                         "verified": statistics.mean(ver.values()), "doubtful": statistics.mean(dbt.values()),
                         "flagged": statistics.mean(flg.values()),
                         "l2": statistics.mean(l2.values()) if l2 else None,
                         "recall": statistics.mean(rec.values()) if rec else None}
        print(f"{prov:<38} {mean_ci(list(ver.values())):>22} {mean_ci(list(dbt.values())):>22} {mean_ci(list(flg.values())):>22} "
              f"{mean_ci(list(l2.values())):>22} {mean_ci(list(rec.values())):>22} {summary[prov]['claims_per_answer']:>14.1f}")
    print("VERIFIED = judge p>=0.85 and numbers found; flagged = UNSUPPORTED (p<0.5) + NUMBER_MISMATCH.")

    if args.audit:
        audit = read_jsonl(args.audit)
        sizes = {s: sum(c["status"] == s for c in claims) for s in STATUSES}
        by_band = defaultdict(list)
        for a in audit:
            by_band[a["judge_status"]].append(a["audit"])
        sizes = {b: sizes[b] for b in by_band}
        print(f"\nAudit of {len(audit)} claims sampled per judge band (pooled over all providers), extrapolated to the {sum(sizes.values())} claims of those bands:")
        print(f"  {'band':<14} {'claims':>7} {'audited':>8} {'TRUE':>6} {'PARTIAL':>8} {'FALSE':>6}")
        for b_, v in by_band.items():
            print(f"  {b_:<14} {sizes[b_]:>7} {len(v):>8} {v.count('TRUE'):>6} {v.count('PARTIAL'):>8} {v.count('FALSE'):>6}")
        summary["audit"] = {}
        for name, outcomes in (("fully supported", {"TRUE"}), ("partly supported (overstated / imprecise)", {"PARTIAL"}),
                               ("false or inverted", {"FALSE"})):
            est, lo, hi = stratified_estimate(sizes, by_band, outcomes)
            summary["audit"][name] = [est, lo, hi]
            print(f"  share of claims {name:<42} {est:.3f}  [95% interval {lo:.3f}, {hi:.3f}]")
        print("  Pooled over all providers.")

        # Per provider: the audit must have been drawn per provider x band (sample_audit.py).
        prov_of = {c["id"]: c["provider"] for c in claims}
        for a in audit:
            a.setdefault("provider", prov_of.get(a["id"]))
        per_prov = {}
        for prov in providers:
            mine = [a for a in audit if a["provider"] == prov]
            if len(mine) < 10:
                continue
            bands = defaultdict(list)
            for a in mine:
                bands[a["judge_status"]].append(a["audit"])
            psizes = {b_: sum(c["provider"] == prov and c["status"] == b_ for c in claims) for b_ in bands}
            per_prov[prov] = (psizes, bands)
        if per_prov:
            print("\nPer provider (audited claims / claims in those bands):")
            for prov, (psizes, bands) in per_prov.items():
                cells = []
                for name, outcomes in (("TRUE", {"TRUE"}), ("PARTIAL", {"PARTIAL"}), ("FALSE", {"FALSE"})):
                    est, lo, hi = stratified_estimate(psizes, bands, outcomes)
                    cells.append(f"{name} {est:.3f} [{lo:.3f}, {hi:.3f}]")
                print(f"  {prov:<38} n={sum(len(v) for v in bands.values()):>3}  " + "   ".join(cells))
                summary["audit"][prov] = {n_: stratified_estimate(psizes, bands, o) for n_, o in
                                          (("TRUE", {"TRUE"}), ("PARTIAL", {"PARTIAL"}), ("FALSE", {"FALSE"}))}
            names = list(per_prov)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    (sa, ba), (sb, bb) = per_prov[names[i]], per_prov[names[j]]
                    d, lo, hi, p_gt = stratified_difference(sa, ba, sb, bb, {"TRUE"})
                    print(f"  fully supported, {names[i][:30]} - {names[j][:30]}: {d:+.3f} [95% interval {lo:+.3f}, {hi:+.3f}], "
                          f"P(first > second) = {p_gt:.3f}")
        else:
            print("  (too few audited claims per provider for a per-provider estimate)")

    print("\nStatus counts per provider (all claims):")
    for prov in providers:
        cs = [c for c in claims if c["provider"] == prov]
        print(f"  {prov:<38} " + "  ".join(f"{s}={sum(c['status'] == s for c in cs)}" for s in STATUSES) + f"  (n={len(cs)})")

    print("\nVERIFIED share by field (all claims):")
    print(f"  {'provider':<38} " + " ".join(f"{f[:12]:>13}" for f in CLAIM_FIELDS))
    for prov in providers:
        cells = []
        for f in CLAIM_FIELDS:
            cs = [c for c in claims if c["provider"] == prov and c["field"] == f]
            cells.append(f"{sum(c['status'] == 'VERIFIED' for c in cs) / len(cs):>13.2f}" if cs else f"{'-':>13}")
        print(f"  {prov:<38} " + " ".join(cells))

    print("\nVERIFIED share by paper (mean over runs):")
    papers_all = sorted({r["source_file"] for r in rows})
    print(f"  {'paper':<12} " + " ".join(f"{p[:14]:>15}" for p in [p.split('(')[0].strip()[:14] for p in providers]))
    for paper in papers_all:
        cells = []
        for prov in providers:
            v = [r["verified"] for r in rows if r["provider"] == prov and r["source_file"] == paper]
            cells.append(f"{statistics.mean(v):>15.2f}" if v else f"{'-':>15}")
        print(f"  {paper[:12]:<12} " + " ".join(cells))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "claim_status.jsonl"), "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps({k: c[k] for k in ("id", "provider", "source_file", "field", "claim", "p", "ungrounded", "status")},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    # Flagged claims with the paper sentences closest to them, for a human to check.
    flagged_path = os.path.join(args.out_dir, "flagged_claims.md")
    with open(flagged_path, "w", encoding="utf-8") as f:
        f.write("# Claims to review\n\nUNSUPPORTED (p < 0.5), NUMBER_MISMATCH (a number is not in the paper) and DOUBTFUL "
                "(0.5 <= p < 0.85). On the gold set no UNSUPPORTED claim was true, but 7 of 24 DOUBTFUL ones were false.\n")
        seen = set()
        for prov in providers:
            f.write(f"\n## {prov}\n")
            for st in ("NUMBER_MISMATCH", "UNSUPPORTED", "DOUBTFUL"):
                group = sorted((c for c in claims if c["provider"] == prov and c["status"] == st),
                               key=lambda c: (c["source_file"], c["p"]))
                f.write(f"\n### {st} ({len(group)})\n")
                for c in group:
                    key = (c["run"], c["source_file"])
                    if key not in indexes:
                        sents = split_sentences(papers[c["run"]][c["source_file"]])
                        indexes[key] = (sents, BM25(sents))
                    sents, bm = indexes[key]
                    ev_txt = retrieve_passages(sents, bm, c["claim"], top_k=2, window=0)
                    extra = f" numbers not in paper: {c['ungrounded']}" if c["ungrounded"] else ""
                    f.write(f"\n- **{c['source_file'][:30]}** [{c['field']}] p={c['p']:.2f}{extra}\n  - claim: {c['claim']}\n")
                    for e in ev_txt:
                        f.write(f"  - paper: {e[:300]}\n")
    print(f"\n[+] {flagged_path}, {os.path.join(args.out_dir, 'claim_status.jsonl')}, {os.path.join(args.out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
