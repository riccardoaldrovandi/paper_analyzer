"""Builds the physics corpus of the fine-tuning set: arXiv papers stratified by subfield and year.

Decided 2026-10-28: physics only, teacher Qwen3.8 with prompt v2.1, every
selected answer reviewed by hand. Target ~150 papers so that ~100+ survive the
automatic filters and the review.

Sampling (deterministic with --seed):
  - 16 physics categories (hep, nucl, gr-qc, astro-ph, cond-mat, quant-ph,
    physics.*), the same number of papers per category;
  - per paper a random year in [--year-from, --year-to] and a random month,
    so the set is not dominated by recent or highly cited work;
  - only the paper's PRIMARY category counts (no cross-lists);
  - metadata pre-filter, before any download: a research article (no review,
    lecture notes, thesis, erratum, proceedings), 4-40 pages and at least one
    figure according to the arXiv comment ("12 pages, 5 figures"), so it fits
    the 128k context and has plots for the figure modes;
  - papers of the gold set and of the earlier test corpora are excluded by
    title, so the evaluation papers never end up in training.
Then the PDF and, when arXiv has one, the LaTeX source are downloaded
(<id>_<title>.pdf and <id>_<title>_source.tar.gz, paired by the pipeline as for
the gold set) and a manifest with the metadata is written.

    python scripts/finetune/build_corpus.py --dry-run          # list the selection only
    python scripts/finetune/build_corpus.py                    # download into corpus_ft/
arXiv API terms: one request every 3 s (--delay); downloads go through the
same rate limit.
"""
import argparse
import glob
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import requests

CATEGORIES = [
    "hep-th", "hep-ph", "hep-ex", "hep-lat", "nucl-th", "gr-qc",
    "astro-ph.CO", "astro-ph.HE", "cond-mat.str-el", "cond-mat.stat-mech",
    "cond-mat.mes-hall", "cond-mat.quant-gas", "quant-ph", "physics.flu-dyn",
    "physics.atom-ph", "physics.optics",
]
API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
HEADERS = {"User-Agent": "paper_analyzer thesis corpus builder (research use)"}
NOT_RESEARCH = re.compile(r"\b(review|lecture|lectures|notes on|primer|tutorial|introduction to|thesis|erratum|corrigendum|"
                          r"proceedings|conference|workshop|white paper|snowmass|book|chapter)\b", re.IGNORECASE)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()[:60]


def excluded_titles() -> set:
    """Titles (normalized prefix) of the gold set and of the earlier test corpora."""
    names = set()
    for d in ("gold_pdfs", "downloaded_pdfs", "gold_latex", "downloaded_latex", "comparison_pdfs"):
        for p in glob.glob(os.path.join(d, "*")):
            base = os.path.basename(p)
            base = re.sub(r"(_source)?\.(pdf|tar\.gz)$", "", base)
            names.add(_norm_title(base.split("_", 1)[1] if "_" in base else base)[:40])
    return names


def pages_figures(comment: str) -> tuple:
    pages = re.search(r"(\d+)\s*(?:pages|pp)", comment or "", re.IGNORECASE)
    figs = re.search(r"(\d+)\s*(?:figures|figs?|plots)\b", comment or "", re.IGNORECASE)
    return (int(pages.group(1)) if pages else None), (int(figs.group(1)) if figs else None)


def query(cat: str, year: int, month: int, delay: float) -> list:
    start = f"{year}{month:02d}01"
    end = f"{year}{month:02d}28"
    params = {"search_query": f"cat:{cat} AND submittedDate:[{start}0000 TO {end}2359]",
              "start": 0, "max_results": 60, "sortBy": "submittedDate", "sortOrder": "ascending"}
    time.sleep(delay)
    r = requests.get(API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", NS):
        prim = e.find("arxiv:primary_category", NS)
        comment = e.find("arxiv:comment", NS)
        out.append({
            "arxiv_id": e.find("a:id", NS).text.rsplit("/abs/", 1)[-1],
            "title": " ".join(e.find("a:title", NS).text.split()),
            "primary_category": prim.get("term") if prim is not None else None,
            "published": e.find("a:published", NS).text[:10],
            "comment": " ".join(comment.text.split()) if comment is not None and comment.text else "",
            "journal_ref": (e.find("arxiv:journal_ref", NS).text if e.find("arxiv:journal_ref", NS) is not None else None),
        })
    return out


def eligible(p: dict, cat: str, excluded: set, min_pages: int, max_pages: int) -> bool:
    pages, figs = pages_figures(p["comment"])
    return (p["primary_category"] == cat and pages is not None and min_pages <= pages <= max_pages
            and figs is not None and figs >= 1 and not NOT_RESEARCH.search(p["title"] + " " + p["comment"])
            and _norm_title(p["title"])[:40] not in excluded)


def download(url: str, path: str, delay: float, want_gzip: bool = False) -> bool:
    if os.path.exists(path):
        return True
    time.sleep(delay)
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
    except requests.RequestException as e:
        print(f"[!] {url}: {e}")
        return False
    if r.status_code != 200:
        print(f"[!] {url}: HTTP {r.status_code}")
        return False
    if want_gzip and r.content[:2] != b"\x1f\x8b":
        return False  # PDF-only submission: arXiv serves the PDF instead of a source tarball
    if not want_gzip and not r.content.startswith(b"%PDF"):
        print(f"[!] {url}: not a PDF")
        return False
    with open(path, "wb") as f:
        f.write(r.content)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--year-from", type=int, default=2010)
    ap.add_argument("--year-to", type=int, default=2025)
    ap.add_argument("--min-pages", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--out", default="corpus_ft")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    excluded = excluded_titles()
    pdf_dir, tex_dir = os.path.join(args.out, "pdfs"), os.path.join(args.out, "latex")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    done = {}
    if os.path.exists(manifest_path):
        for line in open(manifest_path, encoding="utf-8"):
            m = json.loads(line)
            done[m["arxiv_id"]] = m
    print(f"[+] {len(excluded)} excluded titles (gold set and earlier corpora), {len(done)} papers already in the manifest.")

    seen_titles = {_norm_title(m["title"])[:40] for m in done.values()}
    for cat in CATEGORIES:
        have = sum(1 for m in done.values() if m["category"] == cat)
        attempts = 0
        while have < args.per_category and attempts < args.per_category * 6:
            attempts += 1
            year, month = rng.randint(args.year_from, args.year_to), rng.randint(1, 12)
            try:
                cands = [p for p in query(cat, year, month, args.delay)
                         if eligible(p, cat, excluded | seen_titles, args.min_pages, args.max_pages)]
            except Exception as e:
                print(f"[!] {cat} {year}-{month:02d}: query failed ({type(e).__name__}: {e})")
                continue
            if not cands:
                continue
            p = rng.choice(cands)
            pages, figs = pages_figures(p["comment"])
            base = f"{p['arxiv_id'].split('v')[0].replace('/', '-')}_{re.sub(r'[^A-Za-z0-9 ,.()-]+', '', p['title'])[:50].strip()}"
            entry = {"arxiv_id": p["arxiv_id"], "title": p["title"], "category": cat, "published": p["published"],
                     "pages": pages, "figures": figs, "comment": p["comment"], "journal_ref": p["journal_ref"],
                     "pdf": None, "latex": None}
            print(f"[~] {cat:20} {p['published']} {pages:>3}pp {figs:>2}fig  {p['title'][:70]}")
            if not args.dry_run:
                pdf_path = os.path.join(pdf_dir, base + ".pdf")
                if not download(f"https://arxiv.org/pdf/{p['arxiv_id']}", pdf_path, args.delay):
                    continue
                entry["pdf"] = pdf_path
                tex_path = os.path.join(tex_dir, base + "_source.tar.gz")
                if download(f"https://arxiv.org/e-print/{p['arxiv_id']}", tex_path, args.delay, want_gzip=True):
                    entry["latex"] = tex_path
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            done[p["arxiv_id"]] = entry
            seen_titles.add(_norm_title(p["title"])[:40])
            have += 1
        print(f"[+] {cat}: {have}/{args.per_category}")

    n = len(done)
    print(f"\n[+] {n} papers{' (dry run, nothing downloaded)' if args.dry_run else f', manifest: {manifest_path}'}; "
          f"with LaTeX source: {sum(1 for m in done.values() if m.get('latex'))}")


if __name__ == "__main__":
    main()
