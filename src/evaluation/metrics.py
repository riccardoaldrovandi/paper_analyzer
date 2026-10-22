"""
Reference-free quality metrics for teacher-generated training labels.

Two levels, both computed offline from an export-dataset record alone (the
record already holds the exact paper text the model saw and its raw answer, so
no GPU or API call is needed):

- Level 1 (structural gate): is the answer a well-formed, non-degenerate
  instance of SYSTEM_PROMPT's schema (not a copy of the schema's own example
  texts)? Any failure here means "discard".
- Level 2 (grounding score, 0-1): are the numbers, references and datasets it
  cites actually in the paper, and did it keep the PySR formulas out of the
  paper's own fields? Catches hallucinations a valid-looking JSON can hide.
"""
import re
import json
import zlib
from difflib import SequenceMatcher

from src.config import ADVANCED_HEADER as ADVANCED_MARKER, SYSTEM_PROMPT
from src.evaluation import formulas as formula_eq

STRING_FIELDS = ["problem_addressed", "methodology", "experiments", "results", "limitations", "future_work"]
LIST_FIELDS = ["main_contributions", "datasets_used", "visual_analysis_findings"]
# The schema describes relevant_elements as a list but types it as a string,
# so models legitimately produce either.
FLEXIBLE_FIELDS = ["relevant_elements"]
EXPECTED_KEYS = STRING_FIELDS + LIST_FIELDS + FLEXIBLE_FIELDS

# Fields that describe the paper itself: everything except the PySR field.
PAPER_FIELDS = [k for k in EXPECTED_KEYS if k != "visual_analysis_findings"]

EMPTY_VALUES = {"", "not specified", "n/a", "none", "not specified."}
# Placeholder answers meaning "nothing here". With guided JSON a list field
# can't be omitted, so models write e.g. ["None explicitly mentioned"] instead
# of [] (Qwen, run v6). Closed-vocabulary rule: a value is a placeholder only
# if EVERY word is in _FILLER_WORDS and at least one is a negation. A real
# name always has at least one content word ("No Language Left Behind",
# "No-reference IQA", "NoCaps"), so it can never be mistaken for one.
_NEGATION_WORDS = {"none", "no", "not", "n/a", "na", "nil", "nothing", "unknown", "unspecified"}
_FILLER_WORDS = _NEGATION_WORDS | {
    "explicitly", "specifically", "clearly", "directly", "mentioned", "specified", "stated", "provided",
    "used", "available", "reported", "given", "described", "listed", "identified", "applicable",
    "in", "the", "this", "paper", "text", "source", "article", "work", "study", "is", "are", "was", "were",
    "any", "a", "an", "of", "for", "by", "authors", "dataset", "datasets", "data", "benchmark", "benchmarks",
    "specific", "particular", "named", "public", "external", "separate", "standard",
}


def _is_placeholder(text: str) -> bool:
    words = re.findall(r"n/a|[a-z]+", text.lower())
    return bool(words) and all(w in _FILLER_WORDS for w in words) and any(w in _NEGATION_WORDS for w in words)

# Level 1 thresholds. On the first 16 real records (Gemini + free OpenRouter
# models) normal answers sit at 0.00-0.05 repeated 4-grams and 0.39-0.51
# compression ratio; a synthetic looping answer scores 0.98 and 0.03. No real
# degenerate output has been saved yet, so these limits are a first guess with
# margin, to be re-checked once one is.
MAX_REPEATED_NGRAM_RATIO = 0.20
MIN_COMPRESSION_RATIO = 0.25
MAX_EMPTY_FIELD_RATIO = 0.5

# Schema echo: a weak model (Velvet-14B, job 60313868) can return valid JSON
# whose fields are SYSTEM_PROMPT's own example texts ("Brief description of the
# main problem...") or pure filler (". . ."). Neither is caught by the checks
# above. 0.85 similarity catches near-verbatim copies; on the 82 real records
# of runs v3-v9 no field comes close.
ECHO_SIMILARITY = 0.85


def _schema_examples() -> list:
    """The example texts of SYSTEM_PROMPT's JSON schema, normalized."""
    start = SYSTEM_PROMPT.find("{", SYSTEM_PROMPT.find("JSON Schema"))
    end = SYSTEM_PROMPT.find("\n}", start) + 2
    examples = []
    for value in json.loads(SYSTEM_PROMPT[start:end]).values():
        examples.extend(value if isinstance(value, list) else [value])
    return [" ".join(e.lower().split()) for e in examples]


SCHEMA_EXAMPLES = _schema_examples()


def schema_echo(parsed: dict) -> tuple[list, list]:
    """Fields with an item copied from the schema examples, and fields with a filler item (no letter or digit)."""
    echoed, filler = set(), set()
    for k in EXPECTED_KEYS:
        value = parsed.get(k)
        items = value if isinstance(value, list) else [] if value is None else [value]
        for item in items:
            text = " ".join(str(item).lower().split())
            if not text:
                continue
            if not re.search(r"[a-z0-9]", text):
                filler.add(k)
            elif any(SequenceMatcher(None, text, ex).ratio() >= ECHO_SIMILARITY for ex in SCHEMA_EXAMPLES):
                echoed.add(k)
    return sorted(echoed), sorted(filler)


def split_source(user_content: str) -> tuple[str, str]:
    """Separates the paper's own text from the appended ADVANCED PLOT ANALYSIS block."""
    idx = user_content.find(ADVANCED_MARKER)
    if idx == -1:
        return user_content, ""
    return user_content[:idx], user_content[idx:]


def _as_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return "" if value is None else str(value)


def is_empty_value(value) -> bool:
    """Public name of _is_empty, used by main.finalize_answer to clean answers."""
    return _is_empty(value)


def _is_empty(value) -> bool:
    if isinstance(value, list):
        return all(_is_empty(v) for v in value)
    text = _as_text(value).strip().lower()
    return text in EMPTY_VALUES or _is_placeholder(text)


# ---------------------------------------------------------------- Level 1 --

def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    """Share of word n-grams that are repeats of an earlier one (0 = no repetition)."""
    words = text.lower().split()
    if len(words) < n * 5:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams)


def compression_ratio(text: str) -> float:
    """zlib size / raw size: a looping answer compresses far better than prose."""
    raw = text.encode("utf-8")
    if len(raw) < 500:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def level1(raw_answer: str) -> dict:
    result = {"json_ok": False, "missing_keys": [], "extra_keys": [], "type_errors": [], "list_for_string": [],
              "schema_echo": [], "filler_fields": [],
              "empty_field_ratio": None, "repeated_ngram_ratio": None, "compression_ratio": None,
              "passed": False, "reasons": []}
    try:
        parsed = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        result["reasons"].append("invalid JSON")
        return result
    if not isinstance(parsed, dict):
        result["reasons"].append("JSON is not an object")
        return result
    result["json_ok"] = True

    result["missing_keys"] = [k for k in EXPECTED_KEYS if k not in parsed]
    result["extra_keys"] = [k for k in parsed if k not in EXPECTED_KEYS]
    for k in STRING_FIELDS:
        if k not in parsed or isinstance(parsed[k], str):
            continue
        # Several free models answer with a list of strings where the schema
        # wants one string: same content, fixable by normalize_answer().
        if isinstance(parsed[k], list) and all(isinstance(v, str) for v in parsed[k]):
            result["list_for_string"].append(k)
        else:
            result["type_errors"].append(k)
    for k in LIST_FIELDS:
        if k in parsed and not isinstance(parsed[k], list):
            result["type_errors"].append(k)
    for k in FLEXIBLE_FIELDS:
        if k in parsed and not isinstance(parsed[k], (str, list)):
            result["type_errors"].append(k)

    # visual_analysis_findings is legitimately empty for papers without plots.
    present = [k for k in PAPER_FIELDS if k in parsed]
    empty = [k for k in present if _is_empty(parsed[k])]
    result["empty_field_ratio"] = round(len(empty) / len(present), 3) if present else 1.0

    # visual_analysis_findings is excluded: its entries share the same framing
    # sentence by design, which would read as repetition.
    all_text = "\n".join(_as_text(parsed.get(k)) for k in PAPER_FIELDS)
    result["repeated_ngram_ratio"] = round(repeated_ngram_ratio(all_text), 3)
    result["compression_ratio"] = round(compression_ratio(all_text), 3)

    result["schema_echo"], result["filler_fields"] = schema_echo(parsed)

    if result["missing_keys"]:
        result["reasons"].append(f"missing keys: {result['missing_keys']}")
    if result["type_errors"]:
        result["reasons"].append(f"wrong types: {result['type_errors']}")
    if result["empty_field_ratio"] > MAX_EMPTY_FIELD_RATIO:
        result["reasons"].append(f"too many empty fields ({result['empty_field_ratio']})")
    if result["repeated_ngram_ratio"] > MAX_REPEATED_NGRAM_RATIO:
        result["reasons"].append(f"repetitive text (4-gram ratio {result['repeated_ngram_ratio']})")
    if result["compression_ratio"] < MIN_COMPRESSION_RATIO:
        result["reasons"].append(f"repetitive text (compression {result['compression_ratio']})")
    if result["schema_echo"]:
        result["reasons"].append(f"schema text copied instead of an answer: {result['schema_echo']}")
    if result["filler_fields"]:
        result["reasons"].append(f"filler items: {result['filler_fields']}")

    result["passed"] = not result["reasons"]
    return result


# ---------------------------------------------------------------- Level 2 --

_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def _numbers(text: str) -> set:
    """Numbers normalized so '1,000' == '1000' and '0.50' == '0.5'."""
    out = set()
    for m in _NUMBER_RE.findall(text):
        n = m.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def number_grounding(parsed: dict, paper_text: str) -> dict:
    """
    Every number the answer states about the paper should appear in the paper.
    Single digits are skipped: they show up everywhere (section numbers, list
    items), so they'd count as "grounded" without proving anything.
    """
    cited = set()
    for k in PAPER_FIELDS:
        cited |= _numbers(_as_text(parsed.get(k)))
    cited = {n for n in cited if len(n.replace(".", "")) > 1}
    if not cited:
        return {"score": None, "ungrounded": []}
    available = _numbers(paper_text)
    ungrounded = sorted(n for n in cited if n not in available)
    return {"score": round(1 - len(ungrounded) / len(cited), 3), "ungrounded": ungrounded}


_ROMAN = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi", 7: "vii", 8: "viii", 9: "ix", 10: "x",
          11: "xi", 12: "xii", 13: "xiii", 14: "xiv", 15: "xv"}
_REF_RE = re.compile(r"\b(table|tab\.|figure|fig\.|algorithm|alg\.|eqs?\.|equations?)\s*\(?(\d+)\)?((?:\s*(?:-|–|,|and|to)\s*\(?\d+\)?)*)", re.IGNORECASE)


def _ref_found(kind: str, num: int, paper_lower: str) -> bool:
    kind = kind.lower()
    if kind.startswith("eq"):
        return f"({num})" in paper_lower
    if kind.startswith("tab"):
        names = ["table", "tab."]
    elif kind.startswith("fig"):
        names = ["figure", "fig.", "fig"]
    else:
        names = ["algorithm", "alg."]
    variants = [str(num)] + ([_ROMAN[num]] if num in _ROMAN else [])
    return any(re.search(rf"{re.escape(n)}\s*{v}\b", paper_lower) for n in names for v in variants)


def reference_grounding(parsed: dict, paper_text: str) -> dict:
    """Tables / figures / algorithms / equations the answer cites must exist in the paper."""
    paper_lower = paper_text.lower()
    refs = set()
    for k in PAPER_FIELDS:
        for kind, first, rest in _REF_RE.findall(_as_text(parsed.get(k))):
            for num in [first] + re.findall(r"\d+", rest):
                refs.add((kind.lower().rstrip("s."), int(num)))
    if not refs:
        return {"score": None, "ungrounded": []}
    ungrounded = sorted(f"{kind} {num}" for kind, num in refs if not _ref_found(kind, num, paper_lower))
    return {"score": round(1 - len(ungrounded) / len(refs), 3), "ungrounded": ungrounded}


def dataset_grounding(parsed: dict, paper_text: str) -> dict:
    """
    Each dataset name should appear in the paper. Matching is loose on purpose:
    the full name, an acronym in parentheses, or most of its significant words.
    """
    items = parsed.get("datasets_used")
    if not isinstance(items, list):
        return {"score": None, "ungrounded": []}
    items = [str(i) for i in items if not _is_empty(i)]
    if not items:
        return {"score": None, "ungrounded": []}
    paper_lower = paper_text.lower()
    ungrounded = []
    for item in items:
        name = item.lower().strip()
        if name in paper_lower:
            continue
        acronyms = re.findall(r"\(([^)]+)\)", name)
        if any(a.strip() and a.strip() in paper_lower for a in acronyms):
            continue
        words = [w for w in re.findall(r"[a-z0-9\-]+", re.sub(r"\([^)]*\)", "", name)) if len(w) >= 4]
        if words and sum(w in paper_lower for w in words) / len(words) >= 0.5:
            continue
        ungrounded.append(item)
    return {"score": round(1 - len(ungrounded) / len(items), 3), "ungrounded": ungrounded}


_CURVE_RE = re.compile(r"Curve '(.*?)': y = (.*?)(?:, R\^2.*)?$", re.MULTILINE)
# PySR constants with 5+ significant decimals never appear in prose by chance,
# so they're a reliable fingerprint of a formula wherever it gets copied.
_CONSTANT_RE = re.compile(r"\d+\.\d+")


def _fingerprints(equation: str) -> list:
    # sympy prints round constants as e.g. 2.55000000000000: once trailing zeros
    # are stripped those are too short to be distinctive, so they don't count.
    out = []
    for c in _CONSTANT_RE.findall(equation):
        whole, dec = c.split(".")
        dec = dec.rstrip("0")
        if len(dec) >= 5:
            out.append(f"{whole}.{dec}")
    return out


def visual_separation(parsed: dict, advanced_text: str) -> dict:
    """
    The SYSTEM_PROMPT rule, checked mechanically: PySR formulas belong only in
    visual_analysis_findings (coverage), framed as AI approximations (framing),
    and must never leak into the fields describing the paper (leakage).
    """
    findings = parsed.get("visual_analysis_findings")
    findings = findings if isinstance(findings, list) else []
    findings = [str(f) for f in findings if not _is_empty(f)]

    if not advanced_text:
        # No plot analysis was given: any finding is invented.
        return {"score": 0.0 if findings else None, "formulas": 0, "covered": 0, "covered_by_equivalence": 0, "leaked": [],
                "unframed": len(findings), "low_confidence_formulas": 0,
                "note": "findings without ADVANCED section" if findings else ""}

    formulas = _CURVE_RE.findall(advanced_text)
    low_conf = len(re.findall(r"low confidence|not cross-validated", advanced_text))
    paper_fields_text = "\n".join(_as_text(parsed.get(k)) for k in PAPER_FIELDS)
    findings_text = "\n".join(findings)

    covered, by_equivalence, leaked = 0, 0, []
    found = None  # formulas parsed from the findings, only if a fingerprint misses
    for label, equation in formulas:
        constants = _fingerprints(equation)
        if constants:
            hit = any(c in findings_text for c in constants)
            if any(c in paper_fields_text for c in constants):
                leaked.append(label)
        else:
            # A formula with no float constant (e.g. "y = x0*x0") has no fingerprint.
            hit = equation.replace(" ", "") in findings_text.replace(" ", "")
        if not hit:
            # Rounded constants, LaTeX or reordered terms: compare numerically.
            found = formula_eq.candidates(findings_text) if found is None else found
            hit = formula_eq.reported(equation, found)
            by_equivalence += hit
        covered += hit

    framing_words = ("approximation", "not derived", "not stated", "ai-generated", "curve-fit", "curve fit")
    unframed = sum(1 for f in findings if not any(w in f.lower() for w in framing_words))

    if leaked:
        score = 0.0
    elif not formulas:
        # Nothing to check. Scoring this 1.0 rewarded vision models that extract
        # nothing usable (LLaVA-NeXT on Net-proton, 2026-10-08 comparison).
        score = None
    else:
        coverage = covered / len(formulas)
        framing = 1 - unframed / len(findings) if findings else 1.0
        score = round(coverage * framing, 3)
    return {"score": score, "formulas": len(formulas), "covered": covered, "covered_by_equivalence": by_equivalence,
            "leaked": leaked, "unframed": unframed, "low_confidence_formulas": low_conf, "note": ""}


def level2(raw_answer: str, user_content: str) -> dict:
    try:
        parsed = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        return {"score": None}
    if not isinstance(parsed, dict):
        return {"score": None}

    paper_text, advanced_text = split_source(user_content)
    parts = {
        "numbers": number_grounding(parsed, paper_text),
        "references": reference_grounding(parsed, paper_text),
        "datasets": dataset_grounding(parsed, paper_text),
        "visual": visual_separation(parsed, advanced_text),
    }
    # Components with nothing to check (score None) don't count either way.
    scores = [p["score"] for p in parts.values() if p["score"] is not None]
    parts["score"] = round(sum(scores) / len(scores), 3) if scores else None
    # Leaking a PySR formula into the paper's fields is the one error we never
    # want in training data, regardless of how good the rest is.
    if parts["visual"]["leaked"]:
        parts["score"] = 0.0
    return parts


def normalize_answer(raw_answer: str) -> str:
    """
    Rewrites an answer so every STRING_FIELDS value is a single string (a list
    of strings gets joined), giving the student one consistent schema to learn.
    """
    parsed = json.loads(raw_answer)
    for k in STRING_FIELDS:
        if isinstance(parsed.get(k), list):
            parsed[k] = " ".join(str(v).strip() for v in parsed[k])
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _with_llm_findings(raw_answer: str, llm_findings) -> str:
    """
    export-dataset rebuilds visual_analysis_findings deterministically and keeps
    what the teacher wrote in "llm_visual_analysis_findings". Level 2 is scored
    on the teacher's own version, otherwise every model would get a perfect
    visual score and the comparison would lose its only signal on this rule.
    """
    try:
        parsed = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        return raw_answer
    if not isinstance(parsed, dict):
        return raw_answer
    parsed["visual_analysis_findings"] = llm_findings
    return json.dumps(parsed, ensure_ascii=False)


_FIGURE_RE = re.compile(r"^\[Figure: .+?\] \(", re.MULTILINE)


def coverage(parsed: dict, user_content: str, visual: dict) -> dict:
    """
    How much of what the model was given it actually reports. Level 2 measures
    precision only: a terse answer that cites little scores high (Llama +
    LLaVA-NeXT, run v6: L2 0.889 with 2-3 of 10 figures read).
    """
    _, advanced_text = split_source(user_content)
    return {
        "figures_with_data": len(_FIGURE_RE.findall(advanced_text)),
        "formulas_available": visual.get("formulas", 0),
        "formulas_reported": visual.get("covered", 0),
        "empty_fields": [k for k in PAPER_FIELDS if k in parsed and _is_empty(parsed[k])],
    }


def evaluate_record(record: dict, rubric: dict = None) -> dict:
    messages = record.get("messages", [])
    user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
    raw_answer = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
    l2_answer = raw_answer
    if record.get("llm_visual_analysis_findings") is not None:
        l2_answer = _with_llm_findings(raw_answer, record["llm_visual_analysis_findings"])
    l2 = level2(l2_answer, user_content)
    try:
        parsed = json.loads(l2_answer)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    cov = coverage(parsed, user_content, l2.get("visual", {})) if isinstance(parsed, dict) else None
    result = {
        "source_file": record.get("source_file"),
        "provider": record.get("provider"),
        "level1": level1(raw_answer),
        "level2": l2,
        "coverage": cov,
    }
    if rubric is not None:
        # Imported here: rubric.py builds on this module's helpers.
        from src.evaluation.rubric import score_rubric
        # An unparsable answer contains none of the facts.
        result["rubric"] = score_rubric(parsed if isinstance(parsed, dict) else {}, rubric)
    return result
