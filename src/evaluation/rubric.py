"""
Reference-based completeness metric (Level 4, rubric form).

Level 2 measures precision: what the answer states is in the paper. It says
nothing about what the answer leaves out, so a terse answer scores high. A
rubric fixes that without writing a whole gold JSON per paper (the costly part
of a gold set): each held-out paper gets 8-12 atomic, weighted facts a good
answer must contain, checkpoint-style as in PRL-Bench (arXiv 2604.15411).

    Recall_rubric = sum(w_i * found_i) / sum(w_i)    over the checkable facts
    Strict        = 1 if every checkable fact is found

Rubric file: JSONL, one paper per line, matched to records by the basename of
"source_file". Fact types, all checked deterministically:

- name:   one of "aliases" appears as a whole word (case-insensitive)
- number: a number in the answer is within "tolerance" (default 1%) relative
          error of "value", or agrees with it to its stated decimals (the
          PHYSICS-Verified rule); with unit "%", 0.932 and 93.2 are the same
- text:   every group of "keywords" has at least one alternative present
          (case-insensitive substring, so stems like "turbulen" work). A text
          fact without keywords is left for an LLM judge and not scored here.

An optional "field" (string or list) restricts the search to those answer
fields; otherwise every field describing the paper is searched.
"""
import os
import re
import json
import unicodedata

from src.evaluation.metrics import EXPECTED_KEYS, PAPER_FIELDS, _as_text
from src.prompt_v2 import EXPECTED_KEYS_V2

FACT_TYPES = ("name", "number", "text")
DEFAULT_TOLERANCE = 0.01

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def _key(source_file: str) -> str:
    """The paper's name: the PDF and the LaTeX source (<name>_source.tar.gz) of a paper share one rubric."""
    name = os.path.basename(source_file or "")
    for ext in ("_source.tar.gz", ".tar.gz", ".pdf"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def _validate(rubric: dict, line_no: int) -> None:
    where = f"rubric line {line_no} ({rubric.get('source_file', '?')})"
    if not rubric.get("source_file"):
        raise ValueError(f"{where}: missing 'source_file'")
    facts = rubric.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"{where}: 'facts' must be a non-empty list")
    ids = set()
    for fact in facts:
        fid = fact.get("id")
        if not fid or fid in ids:
            raise ValueError(f"{where}: missing or duplicate fact id {fid!r}")
        ids.add(fid)
        if fact.get("type") not in FACT_TYPES:
            raise ValueError(f"{where}, fact {fid}: type must be one of {FACT_TYPES}")
        if not isinstance(fact.get("weight", 1), (int, float)) or fact.get("weight", 1) <= 0:
            raise ValueError(f"{where}, fact {fid}: weight must be a positive number")
        fields = fact.get("field")
        fields = [fields] if isinstance(fields, str) else (fields or [])
        unknown = [f for f in fields if f not in EXPECTED_KEYS and f not in EXPECTED_KEYS_V2]
        if unknown:
            raise ValueError(f"{where}, fact {fid}: unknown field(s) {unknown}")
        if fact["type"] == "name" and not (isinstance(fact.get("aliases"), list) and fact["aliases"]):
            raise ValueError(f"{where}, fact {fid}: a name fact needs a non-empty 'aliases' list")
        if fact["type"] == "number" and not isinstance(fact.get("value"), (int, float)):
            raise ValueError(f"{where}, fact {fid}: a number fact needs a numeric 'value'")


def load_rubrics(path: str) -> dict:
    """Returns {source_file basename: rubric}; raises ValueError on a malformed file."""
    rubrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rubric = json.loads(line)
            _validate(rubric, line_no)
            key = _key(rubric["source_file"])
            if key in rubrics:
                raise ValueError(f"rubric line {line_no}: '{key}' already has a rubric")
            rubrics[key] = rubric
    return rubrics


def rubric_for(rubrics: dict, source_file: str):
    return rubrics.get(_key(source_file)) if rubrics else None


# ---------------------------------------------------------------- matching --

def _numbers(text: str) -> list:
    return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(text)]


def _decimals(value) -> int:
    text = repr(float(value))
    return 0 if text.endswith(".0") else len(text.split(".")[1])


def _number_matches(answer_value: float, fact: dict) -> bool:
    value = float(fact["value"])
    tol = fact.get("tolerance", DEFAULT_TOLERANCE)
    candidates = [answer_value]
    if fact.get("unit") == "%":
        candidates += [answer_value * 100, answer_value / 100]
    for a in candidates:
        if value == 0:
            if abs(a) <= tol:
                return True
        elif abs(a - value) / abs(value) <= tol:
            return True
        # Same value to the reference's own precision (5.43 reported, 5.4 in the rubric).
        if round(a, _decimals(value)) == value:
            return True
    return False


def _name_found(alias: str, text_lower: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", text_lower) is not None


def _keyword_groups(fact: dict) -> list:
    return [[g] if isinstance(g, str) else list(g) for g in fact.get("keywords") or []]


def is_checkable(fact: dict) -> bool:
    return fact["type"] != "text" or bool(_keyword_groups(fact))


def _normalize(text: str) -> str:
    # PDF text keeps typographic ligatures ("\ufb02ux") and Unicode minus / en dash.
    return unicodedata.normalize("NFKC", text).replace("\u2212", "-").replace("\u2013", "-")


def fact_found(fact: dict, text: str) -> bool:
    text = _normalize(text)
    text_lower = text.lower()
    if fact["type"] == "name":
        return any(_name_found(_normalize(a), text_lower) for a in fact["aliases"])
    if fact["type"] == "number":
        return any(_number_matches(n, fact) for n in _numbers(text))
    return all(any(_normalize(k).lower() in text_lower for k in group) for group in _keyword_groups(fact))


def _search_text(parsed: dict, fact: dict, paper_fields: list = PAPER_FIELDS) -> str:
    fields = fact.get("field")
    fields = [fields] if isinstance(fields, str) else (fields or paper_fields)
    return "\n".join(_as_text(parsed.get(k)) for k in fields)


# ----------------------------------------------------------------- scoring --

def score_rubric(parsed: dict, rubric: dict, paper_fields: list = PAPER_FIELDS) -> dict:
    """paper_fields: the answer fields searched by default (v1 or v2 schema, see metrics.fields_for)."""
    found, missing, unscored = [], [], []
    total_w = found_w = 0.0
    by_type = {}
    for fact in rubric["facts"]:
        if not is_checkable(fact):
            unscored.append(fact["id"])
            continue
        w = fact.get("weight", 1)
        hit = fact_found(fact, _search_text(parsed, fact, paper_fields))
        total_w += w
        found_w += w * hit
        (found if hit else missing).append(fact["id"])
        t = by_type.setdefault(fact["type"], [0, 0])
        t[0] += hit
        t[1] += 1
    return {
        "recall": round(found_w / total_w, 3) if total_w else None,
        "strict": bool(total_w) and not missing,
        "found": found,
        "missing": missing,
        "unscored": unscored,
        "by_type": {t: f"{h}/{n}" for t, (h, n) in by_type.items()},
    }


def check_rubric(rubric: dict, paper_text: str) -> list:
    """
    A gold fact must itself be in the paper: flags typos or wrong values in the
    annotation before they distort the metric. Text facts are checked on their
    keywords. Only meaningful on the exact text the model saw (the record's
    user message), which may be truncated.
    """
    return [f"{fact['id']} ({fact['type']}) not found in the paper text"
            for fact in rubric["facts"] if is_checkable(fact) and not fact_found(fact, paper_text)]
