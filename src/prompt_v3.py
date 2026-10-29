"""
Version 3 of the analysis prompt: v1 (src/config.py, SYSTEM_PROMPT_V1) with
its "relevant_elements" field structured as in v2 (src/prompt_v2.py).

v1's "relevant_elements" is one string mixing formulas, algorithms and tables;
v3 keeps every other v1 field unchanged and turns it into an object with v2's
four sub-lists: key_equations, algorithms, tables, figures (same item fields
as v2). A middle ground between v1 (short, free text) and v2 (fully
structured).

Kept as an alternative for later (2026-10-28): it is NOT wired to
PROMPT_VERSION, so nothing uses it yet. To enable it, add "v3" to the switch
in src/config.py and src/llm/schemas.py, and to fields_for()/level1() in
src/evaluation/metrics.py (FIELDS_V3 below gives the field layout).
"""
from src.config import SYSTEM_PROMPT_V1
from src.prompt_v2 import FIELDS_V2, _schema_for

_V1_RELEVANT_ELEMENTS = (
    '  "relevant_elements": "List of crucial formulas, algorithms, or tables mentioned '
    "(e.g., 'Algorithm 1: PPO Update', 'Table 3: Baseline Comparison'), taken only from the paper itself\",\n"
)
_V3_RELEVANT_ELEMENTS = """  "relevant_elements": {
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
    ]
  },
"""
_V1_TYPES = "All values must be strings or arrays of strings in English."
_V3_TYPES = ('All values must be strings or arrays of strings in English, except "relevant_elements": an object with four lists '
             "(key_equations, algorithms, tables, figures), taken only from the paper itself, with at most 8 entries each "
             "and only the elements important for the paper's conclusions.")

assert _V1_RELEVANT_ELEMENTS in SYSTEM_PROMPT_V1 and _V1_TYPES in SYSTEM_PROMPT_V1
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V1.replace(_V1_RELEVANT_ELEMENTS, _V3_RELEVANT_ELEMENTS).replace(_V1_TYPES, _V3_TYPES)

# Field layout (same notation as FIELDS_V2): v1's fields, relevant_elements an object.
FIELDS_V3 = {
    "problem_addressed": "string",
    "main_contributions": "list",
    "methodology": "string",
    "datasets_used": "list",
    "experiments": "string",
    "results": "string",
    "limitations": "string",
    "future_work": "string",
    "relevant_elements": {k: FIELDS_V2[k] for k in ("key_equations", "algorithms", "tables", "figures")},
    "visual_analysis_findings": "list",
}
EXPECTED_KEYS_V3 = list(FIELDS_V3)
PAPER_FIELDS_V3 = [k for k in EXPECTED_KEYS_V3 if k != "visual_analysis_findings"]
ANSWER_SCHEMA_V3 = _schema_for(FIELDS_V3)
