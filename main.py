import argparse
import os
import re
import json
import glob
import hashlib
import subprocess
from datetime import datetime
from src.api.semantic_scholar import fetch_semantic_scholar
from src.api.inspire_hep import fetch_inspire_hep
from src.utils.helpers import sanitize_filename, export_results, process_and_print_result, salvage_without_findings
from src.processing.pdf_extractor import extract_text_from_pdf
from src.processing.latex_extractor import extract_text_from_latex_tarball
from src.llm.gemini_client import analyze_with_gemini
from src.llm.groq_client import analyze_with_groq
from src.llm.openrouter_client import analyze_with_openrouter
from src.llm.local_client import analyze_with_local, fit_to_context, GENERATION_SETTINGS
from src.processing.image_extractor import extract_images_from_pdf, extract_images_from_latex
from src.advanced.digitizer import digitize_plot, digitize_plot_local, looks_synthetic, discrete_points_degenerate, VISION_MODEL_NAME, VISION_GUIDED_JSON
from src.advanced.pysr import discover_formula
from src.config import SYSTEM_PROMPT, PROMPT_VERSION, PDF_DIR, LATEX_DIR, ADVANCED_HEADER
from src.utils.retry import QuotaExceededError
from src.evaluation.report import run_evaluation
from src.evaluation.metrics import fields_for, is_empty_value
from src.prompt_v2 import (FINDING_NOTE, SINGLE_CALL_INSTRUCTIONS, SINGLE_CALL_INSTRUCTIONS_V21, V2_VERSIONS,
                            VERIFY_FIGURES_PROMPT, VERIFY_FIGURES_SCHEMA)
from src.processing import library_extract
from src.processing.page_images import pdf_figure_pages, latex_figure_files

# Fields describing the paper in the prompt this run sends (v1 or v2).
PAPER_FIELDS = fields_for(PROMPT_VERSION)[1]

# Directory holding per-paper resume state for `full-analyze`/`export-dataset`
# (see build_enriched_paper_text / QuotaCheckpointSaved below). Overridable via
# the CHECKPOINT_DIR environment variable: checkpoints store the figure analysis
# produced by one specific vision model, so each model in a comparison run must
# use its own directory or it would silently reuse another model's figures.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "full_analyze_checkpoints")
# KEEP_CHECKPOINTS=1 keeps each paper's figure checkpoint after export-dataset
# completes it, so other text models can reuse the same figure analysis
# (run_dual_model.sbatch SEED_CHECKPOINT_DIR).
KEEP_CHECKPOINTS = os.environ.get("KEEP_CHECKPOINTS", "0") == "1"


class QuotaCheckpointSaved(Exception):
    """
    Raised when OpenRouter itself confirms quota exhaustion (after Gemini
    already had) -- i.e. both free providers are genuinely out of quota, not
    just an unrelated per-image failure. Progress has already been written to
    disk at `checkpoint_path`, so the caller should stop cleanly (no
    traceback) instead of grinding through the rest of a paper's figures
    against a dead quota.
    """
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        super().__init__(f"Progress checkpointed to {checkpoint_path}")


def _paper_base_name(source: str) -> str:
    """Paper file name without its extension (.pdf or .tar.gz)."""
    name = os.path.basename(source)
    for ext in (".tar.gz", ".pdf"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def _checkpoint_path(source: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # The source file's own basename already encodes both the paper identity
    # and whether it's the PDF or the LaTeX (_source.tar.gz) variant.
    return os.path.join(CHECKPOINT_DIR, os.path.basename(source) + ".checkpoint.json")


def _load_checkpoint(checkpoint_path: str) -> dict:
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_checkpoint(checkpoint_path: str, source: str, text_content: str, image_paths: list, advanced_blocks: list, next_index: int) -> None:
    data = {
        "source": source,
        "text_content": text_content,
        "image_paths": image_paths,
        "advanced_blocks": advanced_blocks,
        "next_index": next_index,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _delete_checkpoint(checkpoint_path: str) -> None:
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# Lines written by analyze_figure(): the figure header and one line per fitted curve.
_FIGURE_RE = re.compile(r"^\[Figure: (.+?)\] \(")
_FITTED_CURVE_RE = re.compile(r"^\s*- Curve '(.*?)': y = (.*?)(?:, (R\^2.*))?$")


def _short_figure_name(file_name: str) -> str:
    """'<paper>_page5_crop_plot1.png' -> 'page 5, crop_plot1'; other names (e.g. LaTeX 'fig1.png') unchanged."""
    m = re.search(r"_page(\d+)_(.+)\.\w+$", file_name)
    return f"page {m.group(1)}, {m.group(2)}" if m else file_name


def build_visual_findings(text_content: str) -> list:
    """
    Builds "visual_analysis_findings" directly from the ADVANCED PLOT ANALYSIS
    section: one entry per fitted curve, with the formula copied verbatim and
    framed as an AI approximation. Copying formulas is a purely mechanical task
    that every local teacher got wrong in the 2026-10-08 comparison (Qwen wrote
    "y = ...", Mistral generic text, Llama nothing), and a training target with
    broken formulas would teach the student the same mistake.
    """
    start = text_content.find(ADVANCED_HEADER)
    if start == -1:
        return []
    findings, figure = [], "unknown figure"
    for line in text_content[start:].splitlines():
        header = _FIGURE_RE.match(line)
        if header:
            figure = _short_figure_name(header.group(1))
            continue
        curve = _FITTED_CURVE_RE.match(line)
        if curve:
            label, equation, quality = curve.groups()
            if PROMPT_VERSION in V2_VERSIONS:
                findings.append({"figure": figure, "curve": label, "formula": equation,
                                 "fit_quality": quality or "Not specified", "note": FINDING_NOTE})
                continue
            quality = f"{quality}; " if quality else ""
            findings.append(
                f"AI curve-fit approximation of figure '{figure}', curve '{label}': y = {equation} "
                f"({quality}not derived or stated by the paper's authors)"
            )
    return findings


def _drop_placeholders(parsed: dict) -> None:
    """
    Normalizes "nothing here" answers so the student learns one convention:
    placeholder items in list fields are removed (["None explicitly mentioned"]
    -> []) and placeholder text fields become "Not specified", the value the
    SYSTEM_PROMPT asks for. Uses the same closed-vocabulary rule as `evaluate`,
    so a real name ("No Language Left Behind") is never touched.
    """
    for key in PAPER_FIELDS:
        if key in parsed:
            parsed[key] = _without_placeholders(parsed[key])


def _without_placeholders(value):
    """_drop_placeholders on one value; recurses into v2 objects, and drops list items left all empty."""
    if isinstance(value, dict):
        return {k: _without_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        cleaned = [_without_placeholders(v) for v in value]
        return [v for v in cleaned if not is_empty_value(v)]
    if isinstance(value, str) and value.strip() and is_empty_value(value):
        return "Not specified"
    return value


def finalize_answer(raw_res: str, text_content: str) -> tuple:
    """
    Turns a teacher's raw answer into the final JSON record, or rejects it.
    Returns (final_json, llm_findings); final_json is None if the answer can't
    be used. Two steps:
    - if the JSON is invalid only inside "visual_analysis_findings", the
      record is salvaged without that field (salvage_without_findings) and the
      teacher's version is logged as "[unparseable] ...", which scores 0 in L2;
    - when the input has an ADVANCED PLOT ANALYSIS section, the field is
      replaced by build_visual_findings(), and llm_findings returns what the
      teacher wrote, so `evaluate` can still score the separation rule per model.
    """
    try:
        parsed = json.loads(raw_res)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    llm_findings = None
    if not isinstance(parsed, dict):
        parsed, broken_findings = salvage_without_findings(raw_res, PAPER_FIELDS)
        if parsed is None:
            return None, None
        print("[~] Salvaged an answer whose only invalid JSON was inside visual_analysis_findings.")
        llm_findings = "[unparseable] " + broken_findings[:500]
        parsed["visual_analysis_findings"] = []
    elif not parsed:
        return None, None
    else:
        llm_findings = parsed.get("visual_analysis_findings")

    _drop_placeholders(parsed)

    if ADVANCED_HEADER not in text_content:
        if PROMPT_VERSION == "v2.1":
            # The field belongs to the code in v2.1: with no curve fits it is empty.
            parsed["visual_analysis_findings"] = []
        return json.dumps(parsed, ensure_ascii=False, indent=2), None
    parsed["visual_analysis_findings"] = build_visual_findings(text_content)
    return json.dumps(parsed, ensure_ascii=False, indent=2), llm_findings


def run_llm_providers(text_content: str, model_arg: str) -> bool:
    """
    Runs the requested LLM provider(s) (or all of them) over the given text content
    and prints each resulting structured JSON. Shared between `analyze` and
    `full-analyze` so both commands report results the exact same way.

    Returns True only if every requested provider returned valid, non-empty JSON --
    `full-analyze` uses this to decide whether the paper is really done (and its
    checkpoint can be removed) or whether the final analysis itself hit a dead
    quota and should be retried later.
    """
    providers = ["gemini", "groq", "openrouter"] if model_arg.lower() == "all" else [model_arg.lower()]
    all_succeeded = True

    for provider in providers:
        print(f"\n--- Running analysis with: {provider.upper()} ---")
        if provider == "gemini":
            m_name, raw_res = analyze_with_gemini(text_content)
        elif provider == "groq":
            m_name, raw_res = analyze_with_groq(text_content)
        elif provider == "openrouter":
            m_name, raw_res = analyze_with_openrouter(text_content)
        elif provider == "local":
            m_name, raw_res = analyze_with_local(text_content)
        else:
            continue

        final_res, _ = finalize_answer(raw_res, text_content)

        print(f"\n------------ {provider.capitalize()} ------------")
        print(f"Model used: {m_name}")
        print("Output JSON:")
        process_and_print_result(m_name, final_res if final_res is not None else raw_res)
        print("-" * 40)

        if final_res is None:
            all_succeeded = False

    return all_succeeded


# Below this many digitized points a curve gets no PySR formula at all.
MIN_POINTS_FOR_FIT = 4


def _code_version() -> str:
    """Short git commit of this checkout (+ "-dirty" with local changes), or "unknown" (e.g. rsynced copy without .git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def analyze_figure(image_path: str, iterations: int, use_local_vision: bool = False) -> str:
    """
    Runs the visuo-mathematical pipeline (digitize + symbolic regression) on a
    single extracted figure and returns a plain-text summary block describing it,
    or None if no usable data could be extracted from that image. Any failure on
    an individual figure or curve is logged and skipped rather than aborting the
    whole batch, since one bad chart shouldn't sink the analysis of the rest.

    use_local_vision routes digitization to the local vision-model server
    (--model local) instead of the Gemini/OpenRouter cloud path.
    """
    print(f"\n[~] Analyzing figure: {os.path.basename(image_path)}")
    try:
        plot_data = digitize_plot_local(image_path) if use_local_vision else digitize_plot(image_path)
    except QuotaExceededError:
        raise  # both providers exhausted -- let the caller stop and checkpoint
    except OSError:
        raise  # network failure -- let the caller stop and checkpoint at this figure
    except Exception as e:
        print(f"[!] Skipping {os.path.basename(image_path)}: unexpected error during digitization: {e}")
        return None

    if plot_data is None:
        print(f"[!] Skipping {os.path.basename(image_path)}: could not extract data from this image.")
        return None
    return figure_block(os.path.basename(image_path), plot_data, iterations)


def _curve_readings(X, y) -> str:
    """
    The digitized points of one curve in a line (prompt v2.1): range, values at
    both ends, extremes. A formula alone hides what a reader needs most (where a
    curve starts, where it levels off), and PySR can miss a plateau entirely.
    """
    pts = sorted(zip((float(v) for v in X), (float(v) for v in y)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    i_max, i_min = ys.index(max(ys)), ys.index(min(ys))
    return (f"read points (approx.): x from {xs[0]:.3g} to {xs[-1]:.3g}; y = {ys[0]:.3g} at the left end, "
            f"{ys[-1]:.3g} at the right end; max {ys[i_max]:.3g} at x = {xs[i_max]:.3g}, min {ys[i_min]:.3g} at x = {xs[i_min]:.3g}")


def figure_block(figure_name: str, plot_data: dict, iterations: int) -> str:
    """
    The ADVANCED PLOT ANALYSIS block of one figure from its digitized data
    (digitize_plot() format): PySR on each continuous curve, the points of a
    discrete plot. Shared by the vision pipeline (analyze_figure) and by the
    library-only mode (build_library_enriched_text), so both produce the same
    block for the same data. None if nothing usable is left.
    """
    plot_type = plot_data.get("plot_type", "continuous")
    lines = [f"[Figure: {figure_name}] ({plot_type} plot)"]

    if plot_type == "continuous":
        for series in plot_data.get("series", []):
            label = series.get("label", "curve")
            if len(series["X"]) < MIN_POINTS_FOR_FIT:
                # Any formula "fits" 2-3 points; report the readings themselves.
                points = ", ".join(f"({x:g}, {y:g})" for x, y in zip(series["X"], series["y"]))
                print(f"[!] Skipping curve '{label}': only {len(series['X'])} points, too few for a formula.")
                lines.append(f"  - Curve '{label}': no formula fitted (only {len(series['X'])} points); extracted points: {points}")
                continue
            if looks_synthetic(series["X"], series["y"]):
                # Constant or perfectly regular points are a vision-model
                # fabrication, not data read from the plot: fitting them only
                # yields formulas like "x0" or "2.3e-10" with R^2 = 1.0.
                # Can also be a genuine flat reference line (e.g. a baseline):
                # skipping the fit loses nothing there, so the note stays neutral.
                print(f"[!] Skipping curve '{label}': digitized points are exactly constant or perfectly regular (likely vision-model placeholders), no formula fitted.")
                lines.append(f"  - Curve '{label}': no formula fitted (extracted points are exactly constant or perfectly regular; "
                             f"y from {float(min(series['y'])):g} to {float(max(series['y'])):g}).")
                continue
            try:
                result = discover_formula(series["X"], series["y"], iterations=iterations)
            except Exception as e:
                print(f"[!] Skipping curve '{label}': unexpected error during symbolic regression: {e}")
                continue

            if result.get("r2_holdout") is not None:
                confidence = f", R^2 fit={result['r2']:.3f}, holdout={result['r2_holdout']:.3f}"
                if result["r2_holdout"] < 0.85 or (result["r2"] - result["r2_holdout"]) > 0.15:
                    confidence += " (low confidence, possible overfitting)"
            elif result.get("r2") is not None:
                confidence = f", R^2={result['r2']:.3f} (not cross-validated, too few points)"
            else:
                confidence = ""
            lines.append(f"  - Curve '{label}': y = {result['equation']}{confidence}")
            if PROMPT_VERSION == "v2.1":
                lines.append(f"    {_curve_readings(series['X'], series['y'])}")
    else:
        summary = plot_data.get("summary", "")
        if summary:
            lines.append(f"  Summary: {summary}")
        labels = plot_data.get("labels", [])
        X, y = plot_data.get("X"), plot_data.get("y")
        if X is not None and y is not None and discrete_points_degenerate(X, y):
            print(f"[!] Discarding the data points of {figure_name}: all {len(X)} points are identical (likely vision-model placeholders).")
            lines.append(f"  - Data points discarded: all {len(X)} extracted points were identical ({X[0]:g}, {y[0]:g}), not read from the chart.")
        elif X is not None and y is not None:
            for i, (px, py) in enumerate(zip(X, y)):
                label = labels[i] if i < len(labels) else f"Item {i + 1}"
                lines.append(f"  - {label}: ({px}, {py})")

    if len(lines) == 1:
        # Nothing usable was extracted beyond the header line.
        return None

    return "\n".join(lines)


def build_enriched_paper_text(pdf: str = None, latex: str = None, iterations: int = 20, max_figures: int = 0, use_local_vision: bool = False, figures_dir: str = None) -> str:
    """
    Extracts a paper's text and figures, runs the visuo-mathematical analysis on
    each figure, and returns the text enriched with the ADVANCED PLOT ANALYSIS
    block. This is the exact input both `full-analyze` and `export-dataset` feed
    to the LLM, kept in one place so the two commands can never drift apart.
    Returns None if the source file doesn't exist.

    max_figures caps how many extracted figures actually get analyzed (0 = no
    cap): some papers (e.g. LaTeX sources with dozens of repetitive ablation
    plots) can dump 50+ images, which would otherwise burn through a whole day's
    API quota analyzing a single paper.

    use_local_vision routes every figure's digitization to the local
    vision-model server instead of Gemini/OpenRouter (see analyze_figure).

    figures_dir, if set, skips figure extraction and uses the pre-cropped
    figures in <figures_dir>/<paper file name without extension>/ instead.
    Used for model comparisons (every model must see the exact same figures)
    and on CINECA, where the Gemini-based filter/crop is unavailable and
    extraction would fall back to full-page renders.

    Resumable via a per-paper checkpoint file (see CHECKPOINT_DIR): if a run gets
    cut short by both LLM providers running out of quota, the already-analyzed
    figures are saved to disk and the next call on the same paper picks up where
    it left off instead of re-spending quota re-analyzing figures already done.
    Raises QuotaCheckpointSaved if it has to stop early for this reason.
    """
    source = pdf or latex
    if not source:
        return None
    if not os.path.exists(source):
        print(f"[!] File not found: {source}")
        return None

    checkpoint_path = _checkpoint_path(source)
    checkpoint = _load_checkpoint(checkpoint_path)

    if checkpoint and checkpoint.get("source") == source:
        print(f"[~] Resuming '{os.path.basename(source)}' from checkpoint: figure {checkpoint['next_index'] + 1}/{len(checkpoint['image_paths'])}.")
        text_content = checkpoint["text_content"]
        image_paths = checkpoint["image_paths"]
        advanced_blocks = checkpoint["advanced_blocks"]
        start_index = checkpoint["next_index"]

        # --max-figures also applies on resume: a paper with dozens of figures
        # can otherwise keep re-checkpointing for days without ever finishing,
        # since the full (uncapped) list gets baked into the first checkpoint.
        # Already-analyzed figures (< start_index) are never trimmed away.
        if max_figures > 0 and len(image_paths) > max(max_figures, start_index):
            new_len = max(max_figures, start_index)
            print(f"[!] Capping remaining analysis to figure {new_len}/{len(image_paths)} (--max-figures).")
            image_paths = image_paths[:new_len]
    else:
        if pdf:
            print(f"\n[1/3] Extracting text from PDF: {pdf}")
            text_content = extract_text_from_pdf(pdf)
        else:
            print(f"\n[1/3] Extracting text from LaTeX tarball: {latex}")
            text_content = extract_text_from_latex_tarball(latex)

        if figures_dir:
            paper_dir = os.path.join(figures_dir, _paper_base_name(source))
            image_paths = sorted(
                p for p in glob.glob(os.path.join(paper_dir, "*"))
                if p.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            print(f"\n[2/3] Using {len(image_paths)} pre-cropped figure(s) from '{paper_dir}' (--figures-dir).")
            if not image_paths:
                print(f"[!] No figures found in '{paper_dir}': this paper will be analyzed as text only.")
        else:
            print(f"\n[2/3] Extracting figures...")
            image_paths = extract_images_from_pdf(pdf) if pdf else extract_images_from_latex(latex)

        print(f"[+] Found {len(image_paths)} figure(s) to analyze.")
        if max_figures > 0 and len(image_paths) > max_figures:
            print(f"[!] Capping analysis to the first {max_figures} figure(s) out of {len(image_paths)} (--max-figures).")
            image_paths = image_paths[:max_figures]

        advanced_blocks = []
        start_index = 0

    print(f"\n[3/3] Running visuo-mathematical analysis on each figure...")
    for i in range(start_index, len(image_paths)):
        print(f"\n--- Figure {i + 1}/{len(image_paths)} ---")
        try:
            block = analyze_figure(image_paths[i], iterations, use_local_vision)
        except QuotaExceededError as e:
            # Precise stop condition: this only ever reaches us when OpenRouter
            # itself reported quota exhaustion after Gemini already had --
            # i.e. both free providers are confirmed dead, not just "a few
            # figures in a row happened to fail for other reasons".
            _save_checkpoint(checkpoint_path, source, text_content, image_paths, advanced_blocks, next_index=i)
            print(f"\n[!] Both providers are out of quota ({e}).")
            print(f"[+] Progress saved to '{checkpoint_path}'. Rerun the same command later to resume from figure {i + 1}/{len(image_paths)}.")
            raise QuotaCheckpointSaved(checkpoint_path)
        except OSError as e:
            # A real network failure (DNS, connection reset, ...) mid-run: we
            # never actually got a verdict on this figure, so don't let the loop
            # grind through the rest marking every remaining figure "no data".
            # Stop here and checkpoint AT this figure (not past it) so a rerun
            # retries it once connectivity is back, instead of silently losing
            # it forever.
            _save_checkpoint(checkpoint_path, source, text_content, image_paths, advanced_blocks, next_index=i)
            print(f"\n[!] Vision provider unreachable while analyzing figure {i + 1} (network down or local vision server crashed): {e}")
            print(f"[+] Progress saved to '{checkpoint_path}'. Rerun the same command later to resume from figure {i + 1}/{len(image_paths)} once your connection is back.")
            raise QuotaCheckpointSaved(checkpoint_path)

        if block:
            advanced_blocks.append(block)

    # The figure loop finished (fresh or resumed): checkpoint the completed state
    # too, so if the final LLM call afterwards also hits a dead quota, a rerun
    # skips straight to retrying that instead of redoing this whole figure pass.
    _save_checkpoint(checkpoint_path, source, text_content, image_paths, advanced_blocks, next_index=len(image_paths))

    if advanced_blocks:
        disclaimer = (
            "(The formulas below were derived by an independent AI curve-fitting/symbolic-regression "
            "tool applied to the pixel data of each figure. They are NOT part of the original paper and "
            "were NOT derived or stated by its authors -- report them separately, never as the paper's own results.)"
        )
        text_content += "\n\n" + ADVANCED_HEADER + "\n" + disclaimer + "\n\n" + "\n\n".join(advanced_blocks)
        print(f"\n[+] Enriched paper content with analysis of {len(advanced_blocks)}/{len(image_paths)} figure(s).")
    else:
        print(f"\n[!] No usable figure analysis produced; proceeding with text-only content.")

    return text_content


# ------------------------------------------------- comparison modes B and C --
# export-dataset --mode: "pipeline" (A, build_enriched_paper_text: vision model
# digitizes each figure, then PySR), "single" (B: one multimodal call with the
# figure pages/files attached, no figure analysis of ours) and "libs" (C:
# figure data read by libraries only, then PySR, then a text-only call).

LIBS_DISCLAIMER = (
    "(The formulas below were derived by an independent curve-fitting/symbolic-regression tool applied to "
    "data points read from the vector graphics of each figure (or from the plot data in the LaTeX source). "
    "They are NOT part of the original paper and were NOT derived or stated by its authors -- report them "
    "separately, never as the paper's own results.)"
)
# Single call: images per request, and the tokens kept free for each of them
# when fitting the text (a 1540 px page is ~1.6k tokens for Qwen3.8, ~2.2k for
# Pixtral/Mistral; Gemma 4 uses fewer).
SINGLE_CALL_MAX_IMAGES = int(os.environ.get("SINGLE_CALL_MAX_IMAGES", "12"))
SINGLE_CALL_IMAGE_TOKENS = int(os.environ.get("SINGLE_CALL_IMAGE_TOKENS", "2500"))


def build_library_enriched_text(pdf: str = None, latex: str = None, iterations: int = 20, vector_dir: str = None) -> str:
    """
    Mode C: the paper text, the tables PyMuPDF finds in the PDF, and an
    ADVANCED PLOT ANALYSIS block built from figure data read by
    src/processing/library_extract.py (no model call). Same block format as
    the pipeline, so the final call and the metrics are unchanged. Cached in
    CHECKPOINT_DIR (<file>.libs.json): PySR is not deterministic, and every
    seed of a comparison must see the same input.
    """
    source = pdf or latex
    if not source or not os.path.exists(source):
        print(f"[!] File not found: {source}")
        return None
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cache = os.path.join(CHECKPOINT_DIR, os.path.basename(source) + ".libs.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("source") == source:
            print(f"[~] Library-only analysis of '{os.path.basename(source)}' reused from {cache}.")
            return cached["text"]

    print(f"\n[1/3] Extracting text from {'PDF' if pdf else 'LaTeX tarball'}: {source}")
    text = extract_text_from_pdf(pdf) if pdf else extract_text_from_latex_tarball(latex)
    print("\n[2/3] Reading figure data and tables with libraries only (no model)...")
    plots, tables, notes = library_extract.extract(pdf=pdf, latex=latex, vector_dir=vector_dir)
    for note in notes:
        print(f"[-] {note}")
    print(f"[+] {len(plots)} plot panel(s) read from vector data, {len(tables)} table(s) found.")
    if tables:
        text += "\n\n--- TABLES (extracted from the PDF pages with PyMuPDF) ---\n\n" + "\n\n".join(tables)

    print("\n[3/3] Symbolic regression on the curves read...")
    blocks = []
    for name, plot_data in plots:
        print(f"\n--- {name} ---")
        block = figure_block(name, plot_data, iterations)
        if block:
            blocks.append(block)
    if blocks:
        text += "\n\n" + ADVANCED_HEADER + "\n" + LIBS_DISCLAIMER + "\n\n" + "\n\n".join(blocks)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"source": source, "text": text, "panels": len(plots), "tables": len(tables), "notes": notes}, f, ensure_ascii=False)
    return text


def single_call_inputs(pdf: str = None, latex: str = None, figures_dir: str = None) -> tuple:
    """Mode B: (paper text, image paths) for one multimodal call. PDF: figure crops in figures_dir/<paper>/ if any, then rendered figure pages; LaTeX: its figure files."""
    source = pdf or latex
    if not source or not os.path.exists(source):
        print(f"[!] File not found: {source}")
        return None, []
    base = _paper_base_name(source)
    if pdf:
        text = extract_text_from_pdf(pdf)
        images = pdf_figure_pages(pdf, os.path.join(CHECKPOINT_DIR, "pages", base), SINGLE_CALL_MAX_IMAGES)
        crop_dir = os.path.join(figures_dir, base) if figures_dir else None
        if crop_dir and os.path.isdir(crop_dir):
            # Prompt v2.1: the cropped figures first (a plot is small on a whole
            # page), then the pages for tables and context, within the same cap.
            crops = sorted(p for p in glob.glob(os.path.join(crop_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))
            images = (crops + images)[:SINGLE_CALL_MAX_IMAGES]
    else:
        text = extract_text_from_latex_tarball(latex)
        fig_dir = os.path.join(figures_dir, base) if figures_dir else None
        if not fig_dir or not os.path.isdir(fig_dir):
            images = extract_images_from_latex(latex)
            fig_dir = os.path.dirname(images[0]) if images else None
        images = latex_figure_files(text, fig_dir, SINGLE_CALL_MAX_IMAGES) if fig_dir else []
    print(f"[+] Single multimodal call: {len(images)} image(s) attached to the text.")
    return text, images


# Second pass on the figures (prompt v2.1, modes with images): the model checks
# its own "figures" entries against the attached images. Off by default.
VERIFY_FIGURES = os.environ.get("VERIFY_FIGURES", "0") == "1"


def verify_figures(answer: str, images: list) -> tuple:
    """
    (answer with the corrected "figures" list, the list before the check).
    Any failure keeps the answer unchanged: the pass can only fix, not cost, a paper.
    """
    parsed = json.loads(answer)
    before = parsed.get("figures") or []
    if not before:
        return answer, None
    print(f"\n[~] Verifying {len(before)} figure description(s) against {len(images)} image(s)...")
    try:
        _, raw = analyze_with_local("FIGURES TO CHECK:\n" + json.dumps({"figures": before}, ensure_ascii=False), images=images,
                                    system_prompt=VERIFY_FIGURES_PROMPT, schema=VERIFY_FIGURES_SCHEMA)
        after = json.loads(raw).get("figures")
    except Exception as e:
        print(f"[!] Figure verification failed ({type(e).__name__}: {e}); answer kept unchanged.")
        return answer, None
    if not isinstance(after, list) or len(after) != len(before):
        print(f"[!] Figure verification returned {len(after) if isinstance(after, list) else 'no'} entries for {len(before)}; answer kept unchanged.")
        return answer, None
    changed = sum(a != b for a, b in zip(after, before))
    print(f"[+] Figure verification: {changed}/{len(before)} entr{'y' if changed == 1 else 'ies'} corrected.")
    parsed["figures"] = after
    return json.dumps(parsed, ensure_ascii=False, indent=2), before


def _log_rejected(rejected_path: str, source_path: str, provider, model_name, raw_output: str = None, error: str = None, input_chars: int = None) -> None:
    """
    Appends one failed paper/provider attempt to <output>.rejected.jsonl: the
    raw answer (invalid JSON) or the error that interrupted it. Needed to
    diagnose failures, and the rejection rate is itself a comparison metric.
    """
    entry = {
        "source_file": source_path,
        "provider": model_name or provider,
        "vision_model": VISION_MODEL_NAME if provider == "local" else None,
        "input_chars": input_chars,
        "raw_output": raw_output,
        "error": error,
    }
    with open(rejected_path, "a", encoding="utf-8") as rej_f:
        rej_f.write(_jsonl_line(entry))


def _jsonl_line(obj) -> str:
    """
    One JSONL line. A model can emit a lone UTF-16 surrogate (e.g. "\\ud8ee" in
    its JSON): json.loads accepts it but UTF-8 cannot encode it, which crashed
    the whole export run (job 61447923). It is replaced with "?" instead.
    """
    return json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8") + "\n"


def already_exported_sources(output_path: str) -> set:
    """
    Reads an existing dataset JSONL file (if any) and returns the set of source
    files already exported, so re-running `export-dataset` only processes papers
    that aren't in the dataset yet instead of re-spending API calls on them.
    """
    exported = set()
    if not os.path.exists(output_path):
        return exported

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                exported.add(record.get("source_file"))
            except json.JSONDecodeError:
                continue
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📚 Academic Paper Analyzer Framework & Literature Review Assistant",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Subcommand: fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch papers from academic APIs and download PDF/LaTeX.")
    fetch_parser.add_argument("--api", type=str, choices=["semantic", "inspire", "both"], default="semantic")
    fetch_parser.add_argument("--query", type=str, default="Multi-Agent Reinforcement Learning")
    fetch_parser.add_argument("--limit", type=int, default=3)
    fetch_parser.add_argument("--latex", action="store_true")
    fetch_parser.add_argument("--min-citations", type=int, default=0, help="Minimum number of citations")
    fetch_parser.add_argument("--sort", type=str, default=None, choices=["mostrecent", "mostcited"], help="INSPIRE-only: sort results by date or by citation count instead of relevance")

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local PDF or LaTeX tarball using LLMs.")
    analyze_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    analyze_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")
    analyze_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "local", "all"])

    # Subcommand: images
    images_parser = subparsers.add_parser("images", help="Extract images from a local PDF or LaTeX tarball.")
    images_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    images_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")

    # Subcommand: advanced
    advanced_parser = subparsers.add_parser("advanced", help="Run data extraction (Vision) and symbolic regression (PySR) on a single chart.")
    advanced_parser.add_argument("--image", type=str, required=True, help="Path to the extracted chart image")
    advanced_parser.add_argument("--iterations", type=int, default=20, help="Number of PySR iterations")

    # Subcommand: full-analyze
    full_analyze_parser = subparsers.add_parser(
        "full-analyze",
        help="Full pipeline: extract text + figures, discover formulas/describe each chart, then run LLM structured analysis enriched with the visual findings."
    )
    full_analyze_parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    full_analyze_parser.add_argument("--latex", type=str, help="Path to local LaTeX .tar.gz file")
    full_analyze_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "local", "all"])
    full_analyze_parser.add_argument("--iterations", type=int, default=20, help="Number of PySR iterations per continuous curve")
    full_analyze_parser.add_argument("--max-figures", type=int, default=0, help="Max number of figures to analyze for this paper (0 = no limit)")
    full_analyze_parser.add_argument("--figures-dir", type=str, default=None, help="Use pre-cropped figures from <dir>/<paper name>/ instead of extracting them")

    # Subcommand: export-dataset
    export_dataset_parser = subparsers.add_parser(
        "export-dataset",
        help="Batch-run full-analyze over downloaded papers and export (input, teacher JSON) pairs as a JSONL fine-tuning dataset."
    )
    export_dataset_parser.add_argument("--pdf-dir", type=str, default=PDF_DIR, help="Directory of PDFs to process")
    export_dataset_parser.add_argument("--latex-dir", type=str, default=LATEX_DIR, help="Directory of LaTeX .tar.gz archives to process")
    export_dataset_parser.add_argument("--output", type=str, default="finetune_dataset.jsonl", help="Output JSONL file (appended to, resumable)")
    export_dataset_parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "groq", "openrouter", "local", "all"], help="Teacher model(s) used to generate the training labels")
    export_dataset_parser.add_argument("--iterations", type=int, default=20, help="Number of PySR iterations per continuous curve")
    export_dataset_parser.add_argument("--limit", type=int, default=0, help="Max number of new papers to process this run (0 = no limit)")
    export_dataset_parser.add_argument("--max-figures", type=int, default=0, help="Max number of figures to analyze per paper (0 = no limit)")
    export_dataset_parser.add_argument("--figures-dir", type=str, default=None, help="Use pre-cropped figures from <dir>/<paper name>/ instead of extracting them (same figures for every model in a comparison)")
    export_dataset_parser.add_argument("--mode", type=str, default="pipeline", choices=["pipeline", "single", "libs", "hybrid"],
                                       help="pipeline: vision model digitizes each figure + PySR; single: one multimodal call with the figure pages/files attached (local model only); libs: figure data read by libraries only + PySR")
    export_dataset_parser.add_argument("--vector-dir", type=str, default=None, help="--mode libs, LaTeX: figure files already converted to PDF in <dir>/<paper name>/ (scripts/matrix/prepare_latex_figures.py)")

    # Subcommand: evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Score an export-dataset JSONL (structural + grounding metrics) and optionally write a filtered copy.")
    evaluate_parser.add_argument("--input", type=str, nargs="+", default=["finetune_dataset.jsonl"], help="Dataset JSONL file(s) produced by export-dataset (read-only); same provider + paper in several files = repeated runs")
    evaluate_parser.add_argument("--report", type=str, default="evaluation_report.jsonl", help="Per-record metrics output")
    evaluate_parser.add_argument("--filtered-output", type=str, default=None, help="If set, write records that pass both levels here")
    evaluate_parser.add_argument("--min-l2", type=float, default=0.8, help="Minimum Level 2 grounding score to keep a record")
    evaluate_parser.add_argument("--rubrics", type=str, default=None, help="Gold rubric JSONL: adds rubric recall (completeness) and holds those papers out of the filtered output")
    evaluate_parser.add_argument("--figure-checks", type=str, default=None, help="Gold figure checks JSONL (gold/figure_checks.jsonl): adds the share of figure-reading checks passed (v2 answers)")
    evaluate_parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples for the confidence intervals")

    args = parser.parse_args()

    if args.command == "fetch":
        safe_query_name = sanitize_filename(args.query).replace(" ", "_")
        if args.api in ["semantic", "both"]:
            dataset_s2 = fetch_semantic_scholar(
                query=args.query, 
                limit=args.limit, 
                fetch_latex=args.latex, 
                min_citations=args.min_citations
            )
            export_results(dataset_s2, base_filename=f"{safe_query_name}_semantic")
        if args.api in ["inspire", "both"]:
            dataset_inspire = fetch_inspire_hep(
                query=args.query,
                limit=args.limit,
                fetch_latex=args.latex,
                min_citations=args.min_citations,
                sort_by=args.sort
            )
            export_results(dataset_inspire, base_filename=f"{safe_query_name}_inspire")

    elif args.command == "analyze":
        if not args.pdf and not args.latex:
            print("[!] Please specify either --pdf or --latex file to analyze.")
            exit()

        text_content = ""
        if args.pdf:
            if os.path.exists(args.pdf):
                print(f"\n[~] Extracting text from PDF: {args.pdf}")
                text_content = extract_text_from_pdf(args.pdf)
            else:
                print(f"[!] PDF not found: {args.pdf}")
                exit()
        elif args.latex:
            if os.path.exists(args.latex):
                print(f"\n[~] Extracting text from LaTeX tarball: {args.latex}")
                text_content = extract_text_from_latex_tarball(args.latex)
            else:
                print(f"[!] LaTeX tarball not found: {args.latex}")
                exit()

        run_llm_providers(text_content, args.model)
    elif args.command == "images":
        if not args.pdf and not args.latex:
            print("[!] Specify either --pdf or --latex to extract images.")
            exit()

        extracted = []
        if args.pdf:
            extracted = extract_images_from_pdf(args.pdf)
        elif args.latex:
            extracted = extract_images_from_latex(args.latex)
            
        print(f"\n[+] Operation completed. Found {len(extracted)} images:")
        for img in extracted:
            print(f" -> {img}")
    elif args.command == "advanced":
        if not os.path.exists(args.image):
            print(f"[!] Image not found: {args.image}")
            exit()
            
        print(f"\n--- STARTING VISUO-MATHEMATICAL PIPELINE ---")
        print(f"[1] Extracting structured data with Gemini Vision...")
        plot_data = digitize_plot(args.image)
        
        # Make sure plot_data is not None before going on
        if plot_data is not None:
            plot_type = plot_data.get("plot_type", "continuous")

            print("\n======================================")
            print(" ADVANCED ANALYSIS - FINAL RESULT")
            print("======================================")
            print(f" Image      : {os.path.basename(args.image)}")
            print(f" Plot type  : {plot_type.upper()}")

            if plot_type == "continuous":
                series_list = plot_data.get("series", [])
                print(f"\n[2] Continuous plot detected: {len(series_list)} distinct curve(s). Discovering formulas with PySR...")
                for series in series_list:
                    print(f"\n--- Curve: {series['label']} ({len(series['X'])} points) ---")
                    result = discover_formula(series["X"], series["y"], iterations=args.iterations)
                    print(f" Equation        : {result['equation']}")
                    if result.get("r2_holdout") is not None:
                        gap = result["r2"] - result["r2_holdout"]
                        overfit_warn = " (possible overfitting: don't rely much on this formula)" if (result["r2_holdout"] < 0.85 or gap > 0.15) else ""
                        print(f" R^2 (fit)       : {result['r2']:.4f}")
                        print(f" R^2 (holdout)   : {result['r2_holdout']:.4f}{overfit_warn}")
                    elif result.get("r2") is not None:
                        print(f" R^2             : {result['r2']:.4f} (not validated on a holdout set: too few points)")

            else:
                # Discrete case
                X = plot_data.get("X")
                y = plot_data.get("y")
                print(f"\n[2] Discrete plot detected: PySR disabled.")
                labels = plot_data.get("labels", [])
                summary = plot_data.get("summary", "No summary available.")

                print(f"\n Description: {summary}")
                print("\n Extracted data mapping (X, Y):")
                for i, (px, py) in enumerate(zip(X, y)):
                    label = labels[i] if i < len(labels) else f"Item {i+1}"
                    print(f"  -> {label}: ({px}, {py})")

            print("======================================")
        else:
            print("\n[-] Pipeline stopped: could not extract data from the image (API error or timeout).")
    elif args.command == "full-analyze":
        if not args.pdf and not args.latex:
            print("[!] Please specify either --pdf or --latex file to analyze.")
            exit()

        checkpoint_path = _checkpoint_path(args.pdf or args.latex)
        try:
            text_content = build_enriched_paper_text(pdf=args.pdf, latex=args.latex, iterations=args.iterations, max_figures=args.max_figures, use_local_vision=(args.model.lower() == "local"), figures_dir=args.figures_dir)
        except QuotaCheckpointSaved:
            exit()
        if text_content is None:
            exit()

        if run_llm_providers(text_content, args.model):
            _delete_checkpoint(checkpoint_path)
        else:
            print(f"\n[!] Final LLM analysis failed (quota exhausted?) -- figures are saved in '{checkpoint_path}', rerun the same command later to retry just the final step.")
    elif args.command == "export-dataset":
        pdf_files = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        latex_files = sorted(glob.glob(os.path.join(args.latex_dir, "*.tar.gz")))
        all_sources = [(p, "pdf") for p in pdf_files] + [(p, "latex") for p in latex_files]

        already_done = already_exported_sources(args.output)
        rejected_path = os.path.splitext(args.output)[0] + ".rejected.jsonl"
        if os.environ.get("EXPORT_SKIP_REJECTED") == "1":
            # Resuming a crashed run: papers already rejected keep their
            # rejection instead of being retried until they pass, which
            # would make the rejection rate look better than it is.
            rejected = already_exported_sources(rejected_path)
            print(f"[~] EXPORT_SKIP_REJECTED=1: {len(rejected - already_done)} paper(s) already rejected, not retried.")
            already_done |= rejected
        pending = [(p, kind) for p, kind in all_sources if p not in already_done]
        print(f"[+] {len(all_sources)} paper(s) found, {len(already_done)} already in '{args.output}', {len(pending)} pending.")

        if args.limit > 0:
            pending = pending[:args.limit]
            print(f"[+] Limiting this run to {len(pending)} paper(s) (--limit {args.limit}).")

        providers = ["gemini", "groq", "openrouter"] if args.model.lower() == "all" else [args.model.lower()]
        multimodal = args.mode in ("single", "hybrid")
        if multimodal and PROMPT_VERSION not in V2_VERSIONS:
            print(f"[!] --mode {args.mode} is written for the v2 schema: set PROMPT_VERSION=v2 or v2.1.")
            exit(1)
        if args.mode == "hybrid" and PROMPT_VERSION != "v2.1":
            print("[!] --mode hybrid needs PROMPT_VERSION=v2.1 (its instructions mention both images and library data).")
            exit(1)
        if multimodal and providers != ["local"]:
            print(f"[!] --mode {args.mode} needs --model local (images are sent to the local vLLM server).")
            exit(1)
        final_system_prompt = SYSTEM_PROMPT
        if multimodal:
            final_system_prompt += SINGLE_CALL_INSTRUCTIONS_V21 if PROMPT_VERSION == "v2.1" else SINGLE_CALL_INSTRUCTIONS
        written = 0
        # Same for every record of this run: enough to reproduce or compare runs.
        run_metadata = {
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
            "code_version": _code_version(),
            "pysr_iterations": args.iterations,
            "max_figures": args.max_figures,
            "figures_dir": args.figures_dir,
            "mode": args.mode,
            "verify_figures": VERIFY_FIGURES and multimodal,
        }

        with open(args.output, "a", encoding="utf-8") as out_f:
            for i, (source_path, kind) in enumerate(pending, 1):
                print(f"\n========== Paper {i}/{len(pending)}: {os.path.basename(source_path)} ==========")
                checkpoint_path = _checkpoint_path(source_path)
                images = []
                try:
                    if args.mode == "libs":
                        vdir = os.path.join(args.vector_dir, _paper_base_name(source_path)) if args.vector_dir else None
                        text_content = build_library_enriched_text(**{kind: source_path}, iterations=args.iterations, vector_dir=vdir)
                    elif args.mode == "single":
                        text_content, images = single_call_inputs(**{kind: source_path}, figures_dir=args.figures_dir)
                    elif args.mode == "hybrid":
                        # B + C: the images of the single call, the text and library-read data of libs.
                        vdir = os.path.join(args.vector_dir, _paper_base_name(source_path)) if args.vector_dir else None
                        text_content = build_library_enriched_text(**{kind: source_path}, iterations=args.iterations, vector_dir=vdir)
                        _, images = single_call_inputs(**{kind: source_path}, figures_dir=args.figures_dir)
                    elif kind == "pdf":
                        text_content = build_enriched_paper_text(pdf=source_path, iterations=args.iterations, max_figures=args.max_figures, use_local_vision=(args.model.lower() == "local"), figures_dir=args.figures_dir)
                    else:
                        text_content = build_enriched_paper_text(latex=source_path, iterations=args.iterations, max_figures=args.max_figures, use_local_vision=(args.model.lower() == "local"), figures_dir=args.figures_dir)
                except QuotaCheckpointSaved:
                    print(f"\n[!] Stopping export-dataset early: the vision provider is unavailable (quota exhausted, network down, or local vision server crashed). Progress on '{source_path}' is saved -- rerun the same command later to resume.")
                    break
                except Exception as e:
                    # One broken paper (corrupt PDF, odd LaTeX archive, ...) must
                    # not stop a batch of hundreds: log it and go on.
                    print(f"[!] Skipping {source_path}: unexpected error while preparing the paper: {type(e).__name__}: {e}")
                    _log_rejected(rejected_path, source_path, None, None, error=f"{type(e).__name__}: {e}")
                    continue

                if not text_content:
                    print(f"[!] Skipping {source_path}: no text could be extracted.")
                    continue

                paper_fully_done = True
                for provider in providers:
                    print(f"\n[~] Generating training label with {provider.upper()}...")
                    # The text the teacher actually sees: only the local model has a
                    # hard context limit that can force truncation (fit_to_context).
                    model_input, truncated = text_content, False
                    try:
                        if provider == "gemini":
                            m_name, raw_res = analyze_with_gemini(model_input)
                        elif provider == "groq":
                            m_name, raw_res = analyze_with_groq(model_input)
                        elif provider == "openrouter":
                            m_name, raw_res = analyze_with_openrouter(model_input)
                        elif provider == "local":
                            model_input, truncated = fit_to_context(text_content, reserve_tokens=len(images) * SINGLE_CALL_IMAGE_TOKENS,
                                                                    system_prompt=final_system_prompt)
                            m_name, raw_res = analyze_with_local(model_input, images=images, system_prompt=final_system_prompt)
                        else:
                            continue
                    except Exception as e:
                        print(f"[!] Skipping {provider} on {source_path}: unexpected error during generation: {type(e).__name__}: {e}")
                        paper_fully_done = False
                        _log_rejected(rejected_path, source_path, provider, None, error=f"{type(e).__name__}: {e}", input_chars=len(model_input))
                        continue

                    final_res, llm_findings = finalize_answer(raw_res, model_input)

                    if final_res is None:
                        print(f"[!] Skipping {provider}: empty or invalid JSON output (raw output saved to '{rejected_path}').")
                        paper_fully_done = False
                        # Keep the raw answer: needed to diagnose why a model
                        # fails, and the rejection rate is itself a comparison metric.
                        _log_rejected(rejected_path, source_path, provider, m_name, raw_output=raw_res, input_chars=len(model_input))
                        continue

                    raw_res = final_res
                    figures_before_check = None
                    if provider == "local" and VERIFY_FIGURES and images:
                        raw_res, figures_before_check = verify_figures(raw_res, images)
                    record = {
                        "source_file": source_path,
                        "provider": m_name,
                        # Which model digitized the figures: needed to tell
                        # apart the Qwen/Llama/Mistral pairs in a comparison.
                        "vision_model": VISION_MODEL_NAME if provider == "local" else None,
                        # What the teacher itself wrote before build_visual_findings()
                        # replaced it: `evaluate` scores the separation rule on this.
                        "llm_visual_analysis_findings": llm_findings,
                        # VERIFY_FIGURES=1: the "figures" list before the verification pass.
                        **({"figures_before_check": figures_before_check} if figures_before_check is not None else {}),
                        # The paper body was cut to fit the context (see fit_to_context);
                        # the user message below is exactly what the teacher saw.
                        "input_truncated": truncated,
                        # --mode single: the images attached before the text (same order).
                        **({"images": [os.path.basename(p) for p in images]} if images else {}),
                        "generation": dict(
                            run_metadata,
                            created_at=datetime.now().isoformat(timespec="seconds"),
                            **(dict(GENERATION_SETTINGS, vision_guided_json=VISION_GUIDED_JSON) if provider == "local" else {}),
                        ),
                        "messages": [
                            {"role": "system", "content": final_system_prompt if provider == "local" else SYSTEM_PROMPT},
                            {"role": "user", "content": model_input},
                            {"role": "assistant", "content": raw_res},
                        ],
                    }
                    out_f.write(_jsonl_line(record))
                    out_f.flush()
                    written += 1
                    print(f"[+] Wrote training example ({provider}, {len(model_input)} chars input{', truncated' if truncated else ''}).")

                if paper_fully_done and not KEEP_CHECKPOINTS:
                    _delete_checkpoint(checkpoint_path)
                else:
                    print(f"[~] Keeping checkpoint for '{source_path}' (some provider(s) failed) -- figures won't be redone on the next run.")

        print(f"\n[+] Done. {written} training example(s) appended to '{args.output}'.")
    elif args.command == "evaluate":
        run_evaluation(args.input, args.report, args.filtered_output, args.min_l2, args.rubrics, args.bootstrap,
                       args.figure_checks)
    else:
        parser.print_help()
