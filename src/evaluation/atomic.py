"""
Atomic claim decomposition: prompt, output schema and deterministic sanity checks.

Sentence-level claims are often compound ("A, B and C"), and a judge trained on
short claims scores them badly (docs/METRICS.md 7.2). An LLM splits each
sentence into atomic, self-contained claims; this module holds everything about
that step that needs neither the network nor a GPU.

The decomposer is an LLM too, so it can silently add or correct content. Two
deterministic checks catch that, and a sentence with a flagged atom falls back
to the original sentence instead of feeding the metric an invented claim:

    number_added   a number (2+ digits) of the atom is not in the source sentence
    words_added    fewer than MIN_WORD_COVER of the atom's content words occur in
                   the surrounding text of the same field
"""
import re

SYSTEM_PROMPT = (
    "You split one sentence from a summary of a scientific paper into atomic claims.\n"
    "Rules:\n"
    "1. Each claim states exactly one fact and can be checked on its own.\n"
    "2. Keep every number, unit, symbol, formula and name exactly as written.\n"
    "3. Make each claim self-contained: replace pronouns and phrases such as 'the authors', 'they', "
    "'this method' with what they refer to, using ONLY the surrounding text given to you.\n"
    "4. Do NOT add information, do NOT correct, interpret, soften or strengthen anything, even if the "
    "sentence looks wrong. Do not drop a fact either.\n"
    "5. If the sentence already states a single fact, return it unchanged as the only claim.\n"
    'Answer with JSON: {"claims": ["...", "..."]}.'
)

SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
}

MIN_ATOM_CHARS = 20
MIN_WORD_COVER = 0.7
_WORD = re.compile(r"[^\W\d_]{4,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def build_messages(sentence: str, context: str) -> list:
    user = f"Surrounding text (for resolving references only):\n{context}\n\nSentence to split:\n{sentence}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _stems(text: str) -> set:
    return {w[:5] for w in _WORD.findall(text.lower())}


def _numbers(text: str) -> set:
    return {n for n in _NUMBER.findall(text) if len(n.replace(".", "")) > 1}


def check_atoms(sentence: str, context: str, atoms: list) -> tuple:
    """
    Returns (atoms to use, reason for falling back or None). The atoms are the
    cleaned, de-duplicated model output; if there is none, or one fails a check,
    the sentence itself is returned as the only claim, with the reason.
    """
    cleaned, seen = [], set()
    for a in atoms if isinstance(atoms, list) else []:
        a = " ".join(str(a).split())
        if len(a) >= MIN_ATOM_CHARS and a.lower() not in seen:
            seen.add(a.lower())
            cleaned.append(a)
    if not cleaned:
        return [sentence], "no_valid_atoms"
    allowed_numbers = _numbers(sentence)
    context_stems = _stems(context) | _stems(sentence)
    for a in cleaned:
        if _numbers(a) - allowed_numbers:
            return [sentence], "number_added"
        stems = _stems(a)
        if stems and len(stems & context_stems) / len(stems) < MIN_WORD_COVER:
            return [sentence], "words_added"
    return cleaned, None
