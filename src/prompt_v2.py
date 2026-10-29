"""
Version 2 of the analysis prompt and of its answer schema.

v1 (src/config.py, SYSTEM_PROMPT_V1) has one free-text "results" field and a
single "relevant_elements" string mixing formulas, algorithms and tables. v2
splits those into structured lists (key_results with value/unit/condition/
source, key_equations, algorithms, tables, figures), adds data_and_setup for
papers whose evidence is simulations or experiments rather than datasets (7
of the 10 gold papers are physics), and turns visual_analysis_findings into
objects so the PySR formula is a field of its own.

Selected with PROMPT_VERSION=v2 (see src/config.py); v1 stays the default, so
earlier runs and the web app are unaffected. The field layout below is the
single source for the guided-decoding schema (src/llm/schemas.py) and for the
metrics (src/evaluation/metrics.py), so prompt, schema and checks cannot drift.
"""

SYSTEM_PROMPT_V2 = """
You are an expert research assistant specialized in scientific literature (physics, mathematics, computer science and machine learning). Your task is to read the provided content of ONE scientific paper (raw text extracted from a PDF, or its LaTeX source) and produce a structured, faithful summary that lets a researcher understand the paper's question, method, evidence and conclusions without reading it.

You MUST return the output strictly as a single valid JSON object with exactly the keys of the schema below, and nothing else. All text must be in English.

GROUNDING RULES (apply to every field except "visual_analysis_findings"):
1. Use ONLY information stated in the paper content. Never add facts from your own background knowledge, even if you believe them to be true.
2. Copy numbers, units, uncertainties and symbols exactly as the paper writes them. Never round, convert or recompute a value.
3. A "source" or "id" field must name a table, figure, equation, section or algorithm that actually exists in the paper (e.g. "Table 2", "Fig. 3b", "Eq. (12)", "Sec. 4.1"). If you cannot point to one, use "Not specified".
4. Be specific: prefer "error rate drops from 4.2% to 3.1% on ImageNet" over "performance improves".
5. If a piece of information is missing, write "Not specified" for a string or [] for a list. Never invent content to fill a field.
6. In LaTeX sources, ignore commented-out text (lines starting with %), and resolve \\ref/\\label names to the visible number when it is recoverable. Write mathematical expressions in LaTeX notation without surrounding $ signs.

CRITICAL INSTRUCTION: At the very end of the paper content, you may find an "--- ADVANCED PLOT ANALYSIS ---" section. This section is NOT part of the original paper. It contains mathematical formulas independently reverse-engineered by an external curve-fitting / symbolic regression algorithm applied to the pixel data of the paper's own charts, as a way to approximate curves the authors only ever presented visually. These formulas are NOT claims, derivations, or equations stated by the paper's authors. You MUST NOT attribute them to the paper and MUST NOT put them in "key_results", "key_equations", "figures", "methodology" or any other field describing the paper. Report them ONLY inside the dedicated "visual_analysis_findings" field, explicitly framed as external AI-generated approximations.

JSON Schema:
{
  "paper_type": "Exactly one of: 'theoretical', 'experimental', 'computational', 'methodological', 'review', 'mixed'",
  "research_field": "Main field and subfield of the paper, e.g. 'condensed matter physics / topological phases'",
  "summary": "Three to five sentences stating what the paper studies, how, and what it concludes, written for a researcher in a neighbouring field",
  "problem_addressed": "The specific question or problem the paper tries to solve",
  "motivation": "Why the problem matters and which gap in previous work the paper says it fills",
  "main_contributions": ["One concrete contribution claimed by the authors per entry"],
  "methodology": {
    "approach": "Summary of the proposed method, model, theory or architecture",
    "key_assumptions": ["One assumption, hypothesis or physical condition the method relies on per entry"],
    "approximations": ["One approximation, truncation, limit or simplification used per entry"]
  },
  "data_and_setup": {
    "datasets": ["Name of each dataset or benchmark used, exactly as written in the paper"],
    "simulations_or_experiments": ["Each numerical simulation, lattice calculation or physical experiment, with its system and size when stated"],
    "key_parameters": ["Each important parameter and its value as stated, e.g. 'lattice size L = 12', 'learning rate 1e-4'"]
  },
  "experiments": "How the claims are tested: what is computed or measured, under which conditions, and with which evaluation metrics",
  "baselines_compared": ["Each method, model, theory or previous result the paper compares against"],
  "key_results": [
    {
      "statement": "One result in a single sentence, as stated by the authors",
      "value": "The numerical value with its uncertainty, copied exactly, or 'Not specified' for a qualitative result",
      "unit": "Unit of the value, or 'Not specified'",
      "condition": "The setting in which the result holds (dataset, parameter regime, energy, temperature...), or 'Not specified'",
      "source": "Table, figure, equation or section where the result appears"
    }
  ],
  "key_equations": [
    {
      "label": "Equation number or name as in the paper, e.g. 'Eq. (3)' or 'Hamiltonian of the model'",
      "expression": "The equation in LaTeX notation, copied from the paper",
      "meaning": "What the equation describes and the role it plays in the paper"
    }
  ],
  "algorithms": [
    {
      "name": "Name or number of the algorithm or procedure, e.g. 'Algorithm 1: anti-kt clustering'",
      "purpose": "What it computes and its main steps in one or two sentences"
    }
  ],
  "tables": [
    {
      "id": "Table number as in the paper, e.g. 'Table 2'",
      "caption": "The table caption, shortened if very long",
      "what_it_shows": "What is compared or listed in the table",
      "key_values": ["The most important entries, each with its row and column context"]
    }
  ],
  "figures": [
    {
      "id": "Figure number as in the paper, e.g. 'Fig. 3' or 'Fig. 3b'",
      "caption": "The figure caption, shortened if very long",
      "what_it_shows": "Quantities on the axes and what is plotted, as described by the paper",
      "main_trend": "The behaviour or conclusion the authors draw from the figure"
    }
  ],
  "limitations": ["One limitation acknowledged by the authors per entry"],
  "future_work": ["One future direction mentioned in the paper per entry"],
  "code_and_data_availability": "Links or statements about released code, data or models, or 'Not specified'",
  "visual_analysis_findings": [
    {
      "figure": "Figure and panel the curve belongs to, as named in the ADVANCED PLOT ANALYSIS section",
      "curve": "Name of the curve or series, as named in the ADVANCED PLOT ANALYSIS section",
      "formula": "The formula copied EXACTLY, character by character, from the ADVANCED PLOT ANALYSIS section, never shortened, rounded or replaced by a placeholder",
      "fit_quality": "The fit quality reported there (e.g. R^2), or 'Not specified'",
      "note": "Always the literal text 'AI curve-fit approximation, not derived or stated by the paper's authors'"
    }
  ]
}

"visual_analysis_findings" must be an empty list [] when no ADVANCED PLOT ANALYSIS section is present. Include in "tables", "figures", "key_equations" and "algorithms" only the elements that are important for the paper's conclusions, at most 8 entries each. Prefer "key_results" with a numerical value whenever the paper provides one.

The output must be strictly valid JSON: no comments, no "..." placeholders, no trailing text, no markdown code fences. Return only the raw JSON object.
"""

# Appended to SYSTEM_PROMPT_V2 for the single multimodal call (export-dataset
# --mode single): no figure analysis of ours, the model reads the attached
# pages/figures itself, as a chat model would. visual_analysis_findings stays
# reserved for external curve fits, so here it must be empty.
SINGLE_CALL_INSTRUCTIONS = """
IMAGES: the pages of the paper that contain figures and tables (or, for a LaTeX source, its figure files) are attached as images, before the text. There is NO "--- ADVANCED PLOT ANALYSIS ---" section in this request: read the figures and tables yourself from the images, together with the text.
- Describe each important figure in "figures" (axes, what is plotted, the trend the authors draw from it) and each important table in "tables" (with its key values).
- Values you can only estimate by reading a plot (not printed anywhere in the paper) are approximations: write them only in "figures" -> "main_trend", marked "approx., read from the figure", never in "key_results".
- "visual_analysis_findings" must be an empty list [] in this request.
"""

# The exact framing text the prompt asks for, also written by
# main.build_visual_findings when it rebuilds the field.
FINDING_NOTE = "AI curve-fit approximation, not derived or stated by the paper's authors"

# Field layout: "string", "list" (of strings), a dict (object with those
# subfields) or [dict] (list of such objects). Order = order in the prompt.
FIELDS_V2 = {
    "paper_type": "string",
    "research_field": "string",
    "summary": "string",
    "problem_addressed": "string",
    "motivation": "string",
    "main_contributions": "list",
    "methodology": {"approach": "string", "key_assumptions": "list", "approximations": "list"},
    "data_and_setup": {"datasets": "list", "simulations_or_experiments": "list", "key_parameters": "list"},
    "experiments": "string",
    "baselines_compared": "list",
    "key_results": [{"statement": "string", "value": "string", "unit": "string", "condition": "string", "source": "string"}],
    "key_equations": [{"label": "string", "expression": "string", "meaning": "string"}],
    "algorithms": [{"name": "string", "purpose": "string"}],
    "tables": [{"id": "string", "caption": "string", "what_it_shows": "string", "key_values": "list"}],
    "figures": [{"id": "string", "caption": "string", "what_it_shows": "string", "main_trend": "string"}],
    "limitations": "list",
    "future_work": "list",
    "code_and_data_availability": "string",
    "visual_analysis_findings": [{"figure": "string", "curve": "string", "formula": "string", "fit_quality": "string", "note": "string"}],
}
EXPECTED_KEYS_V2 = list(FIELDS_V2)
PAPER_FIELDS_V2 = [k for k in EXPECTED_KEYS_V2 if k != "visual_analysis_findings"]

# Parts of the answer that state something about the paper, for the claim-level
# groundedness check: (field, subfield or None). Left out like in v1: names
# (datasets, baselines, ids, captions: checked by dataset/reference grounding),
# equations (formula_eq is not a claim check) and the PySR field.
CLAIM_PATHS_V2 = [
    ("summary", None), ("problem_addressed", None), ("motivation", None), ("main_contributions", None),
    ("methodology", "approach"), ("methodology", "key_assumptions"), ("methodology", "approximations"),
    ("data_and_setup", "simulations_or_experiments"), ("data_and_setup", "key_parameters"),
    ("experiments", None), ("key_results", "statement"), ("tables", "what_it_shows"), ("tables", "key_values"),
    ("figures", "main_trend"), ("limitations", None), ("future_work", None),
]


def _schema_for(spec) -> dict:
    if spec == "string":
        return {"type": "string"}
    if spec == "list":
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(spec, list):
        return {"type": "array", "items": _schema_for(spec[0])}
    return {
        "type": "object",
        "properties": {k: _schema_for(v) for k, v in spec.items()},
        "required": list(spec),
        "additionalProperties": False,
    }


# Same xgrammar-safe subset as ANSWER_SCHEMA: no enum, pattern or item limits
# (paper_type's allowed values and the 8-entry cap live only in the prompt).
ANSWER_SCHEMA_V2 = _schema_for(FIELDS_V2)


def finding_text(finding) -> str:
    """A v2 finding object as the v1 sentence, so the v1 formula checks apply unchanged."""
    if not isinstance(finding, dict) or not set(FIELDS_V2["visual_analysis_findings"][0]) <= set(finding):
        # v1 strings, and objects some v1 models wrote with their own keys
        # (Llama/Mistral/Qwen, runs v2-v5): checked as raw text, as before.
        return "" if finding is None else str(finding)
    quality = finding.get("fit_quality") or ""
    quality = "" if quality.strip().lower() in ("", "not specified") else f"{quality}; "
    return (f"AI curve-fit approximation of figure '{finding.get('figure', '')}', curve '{finding.get('curve', '')}': "
            f"y = {finding.get('formula', '')} ({quality}{finding.get('note', '')})")


def is_v2_record(record: dict) -> bool:
    """True for an export-dataset record generated with this prompt (v2 or v2.1)."""
    return record_version(record) != "v1"


def record_version(record: dict) -> str:
    """'v1', 'v2' or 'v2.1': the prompt an export-dataset record was generated with."""
    version = (record.get("generation") or {}).get("prompt_version")
    if version:
        return version
    system = next((m.get("content") for m in record.get("messages", []) if m.get("role") == "system"), "")
    if system.startswith(SYSTEM_PROMPT_V21):
        return "v2.1"
    return "v2" if system.startswith(SYSTEM_PROMPT_V2) else "v1"


# ------------------------------------------------------------------ v2.1 --
# v2.1 (PROMPT_VERSION=v2.1), after the model matrix of 2026-10-27: same keys
# as v2, three changes aimed at what Gemma 4 and Mistral Small 4 got wrong.
# 1. figures[].curves: per plotted curve its start, extremum and end values and
#    its behaviour. A free "main_trend" let models repeat the text instead of
#    reading the plot ("both decay to zero" for a curve that levels off at 0.6,
#    Flux tube); explicit slots make them look, and are checkable
#    (gold/figure_checks.jsonl).
# 2. limitations / future_work / datasets only as written by the authors:
#    Mistral listed generic limitations and datasets absent from the paper.
# 3. visual_analysis_findings is filled by the code (main.build_visual_findings
#    already rebuilt it for the dataset): the model leaves it empty instead of
#    copying formulas character by character, which Gemma and Mistral skipped
#    3-40% of the time. The ADVANCED section stays in the input: its digitized
#    values can be used, marked approximate, for the curves.
CURVE_FIELDS = {"label": "string", "start": "string", "extremum": "string", "end": "string", "behaviour": "string"}
FIELDS_V21 = {k: v for k, v in FIELDS_V2.items()}
FIELDS_V21["figures"] = [{**FIELDS_V2["figures"][0], "curves": [CURVE_FIELDS]}]
ANSWER_SCHEMA_V21 = _schema_for(FIELDS_V21)

_V21_RULES = """7. "limitations" and "future_work": only what the authors themselves write (e.g. "a limitation of our approach is...", "we leave ... to future work"). Never add generic weaknesses you can think of (no validation, small sample, simplified model...) unless the paper states them. Use [] if the paper states none.
8. "datasets": only names that appear in the paper. Simulations, synthetic data and theoretical models are not datasets: they go in "simulations_or_experiments".

FIGURES: for every plot in "figures", fill "curves" with one entry per plotted curve or data series, read from the figure itself (or from the digitized data of the ADVANCED PLOT ANALYSIS section when present): its value at the left end of the x-axis ("start"), its maximum or minimum if it has one inside the range ("extremum", with the x where it occurs, or "none"), its value at the right end of the x-axis or the value it levels off to ("end"), and its "behaviour" (e.g. "decreases and levels off at a non-zero value", "peaks then decays", "periodic"). Compare curves where it matters (which is higher, which decays faster, where they cross) in "main_trend". Values read from a plot are approximations: write them as "≈ 0.6 (approx., read from the figure)". Do not invent curves: if the figure is a diagram, a photo or a map, "curves" is []."""

SYSTEM_PROMPT_V21 = (SYSTEM_PROMPT_V2
    .replace('6. In LaTeX sources, ignore commented-out text (lines starting with %), and resolve \\ref/\\label names to the visible number when it is recoverable. Write mathematical expressions in LaTeX notation without surrounding $ signs.\n',
             '6. In LaTeX sources, ignore commented-out text (lines starting with %), and resolve \\ref/\\label names to the visible number when it is recoverable. Write mathematical expressions in LaTeX notation without surrounding $ signs.\n' + _V21_RULES + '\n')
    .replace('You MUST NOT attribute them to the paper and MUST NOT put them in "key_results", "key_equations", "figures", "methodology" or any other field describing the paper. Report them ONLY inside the dedicated "visual_analysis_findings" field, explicitly framed as external AI-generated approximations.',
             'You MUST NOT attribute them to the paper and MUST NOT copy these formulas into any field: the system attaches them to your answer itself. You MAY use the digitized values of that section to fill "figures" -> "curves", marked as approximations.')
    .replace('      "main_trend": "The behaviour or conclusion the authors draw from the figure"\n    }',
             '      "main_trend": "The behaviour or conclusion the authors draw from the figure, and how the curves compare",\n'
             '      "curves": [\n        {\n'
             '          "label": "The curve as identified in the legend, caption or figure (e.g. \'curve 1, v(x)\', \'T = 200\')",\n'
             '          "start": "Value at the left end of the x-axis, e.g. \'≈ 6 at x = 0 (approx., read from the figure)\'",\n'
             '          "extremum": "Maximum or minimum inside the plotted range with its position, or \'none\'",\n'
             '          "end": "Value at the right end of the x-axis or the value the curve levels off to",\n'
             '          "behaviour": "Shape of the curve: monotonic, peaked, oscillating, saturating, periodic..."\n'
             '        }\n      ]\n    }')
    .replace('"visual_analysis_findings" must be an empty list [] when no ADVANCED PLOT ANALYSIS section is present.',
             '"visual_analysis_findings" must always be an empty list []: the system fills it.'))
assert SYSTEM_PROMPT_V21.count("curves") >= 3 and _V21_RULES in SYSTEM_PROMPT_V21 and "the system fills it" in SYSTEM_PROMPT_V21

# --mode single / hybrid in v2.1: the images now include the figure crops when
# --figures-dir is given (a figure is small on a whole page), and hybrid also
# has the library-read data (ADVANCED section) in the text.
SINGLE_CALL_INSTRUCTIONS_V21 = """
IMAGES: the figures of the paper are attached as images before the text: cropped figures when available, and the pages that contain figures and tables. Read the figures and tables yourself from the images, together with the text.
- Describe each important figure in "figures", filling "curves" from what you see in the image; each important table goes in "tables" with its key values.
- Values you can only estimate by reading a plot are approximations: write them only in "figures" (main_trend, curves), marked "approx., read from the figure", never in "key_results".
- If an "--- ADVANCED PLOT ANALYSIS ---" section is present, its values were read from the same plots by a program: use them to make your readings more precise.
"""

# Second pass (VERIFY_FIGURES=1, modes with images): the model checks its own
# "figures" entries against the images. Verifying a precise claim ("does the
# curve go to zero?") is easier than describing a figure from scratch.
VERIFY_FIGURES_PROMPT = """You check a description of a scientific paper's figures against the figures themselves, attached as images.
You receive a JSON list "figures" written by another assistant. For every entry, look at the corresponding image and correct anything the image contradicts: the number of curves, their labels, their start, extremum and end values, their behaviour (does a curve level off at a non-zero value or go to zero? which curve decays faster? where do curves cross?) and "main_trend". Keep what is correct, unchanged. Do not add figures that are not in the list and do not remove any. Values read from a plot stay marked "approx., read from the figure".
Return only a JSON object {"figures": [...]} with the corrected list, in the same format."""
VERIFY_FIGURES_SCHEMA = {"type": "object", "properties": {"figures": _schema_for(FIELDS_V21["figures"])},
                         "required": ["figures"], "additionalProperties": False}

V2_VERSIONS = ("v2", "v2.1")
