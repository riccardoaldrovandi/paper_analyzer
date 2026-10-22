"""
Numeric equivalence of a reported formula and a PySR formula.

The constant fingerprints of metrics.visual_separation only recognize a PySR
formula copied with its full-precision constants. Models also round them
("exp(-0.7072*x)*1.2943" for exp(x0*(-0.7072197))*1.2942798), rewrite them in
LaTeX ("\\frac{4.96}{-x_{0} + \\exp{(x_{0}-0.19)}}") or reorder the terms. Here
both expressions are parsed with sympy and compared on a fixed grid of x
values, with the PHYSICS-Verified tolerance: equivalent if they agree within
1% relative error on at least 90% of the points where both are real and
finite.
"""
import re

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

TOLERANCE = 0.01
MIN_AGREEMENT = 0.9
MIN_POINTS = 10

_X = sp.Symbol("x0")
# Figures span very different x ranges (0-6 for a flux-tube profile, 7.7-200
# GeV for a beam-energy scan): a log-spaced grid covers both.
_GRID = np.concatenate([np.geomspace(0.05, 300, 40), -np.geomspace(0.05, 300, 10)])

_LATEX = [
    (r"\\left|\\right", ""),
    (r"\\text\{([^{}]*)\}", r"\1"),
    (r"\\ast|\\cdot|\\times|\\\*", "*"),
    (r"\\exp", "exp"), (r"\\log|\\ln", "log"), (r"\\sqrt", "sqrt"),
    (r"x_\{?0\}?", "x0"),
    (r"\*\*|\^", "**"),
    ("\u2212", "-"),
]
# Characters a formula can be made of; the reported text around it is not.
_MATH = re.compile(r"(?:[\d.+\-*/()\s]|exp|log|sqrt|sin|cos|tanh|tan|abs|pi|x0|x)+")
# A formula starts after "y =" or as a dict/JSON string value ("formula": "...").
_START = re.compile(r"y\s*=\s*|:\s*[\"']\s*")


def _brace_group(text: str, start: int):
    """(content, end) of the balanced {...} group opening at text[start], or None."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i], i + 1
    return None


def _expand_frac(text: str) -> str:
    """\\frac{a}{b} -> ((a)/(b)), braces nested at any depth."""
    while (i := text.find("\\frac")) != -1:
        num = _brace_group(text, i + 5)
        den = num and _brace_group(text, num[1])
        if not den:
            # Malformed: drop the command so the loop ends; parsing will then fail.
            text = text[:i] + text[i + 5:]
            continue
        text = f"{text[:i]}(({_expand_frac(num[0])})/({_expand_frac(den[0])})){text[den[1]:]}"
    return text


def _normalize(text: str) -> str:
    for pattern, repl in _LATEX:
        text = re.sub(pattern, repl, text)
    return _expand_frac(text).replace("{", "(").replace("}", ")")


def parse(expr: str):
    """sympy expression in x0 (x is accepted as x0), or None."""
    try:
        e = parse_expr(_normalize(expr), transformations=standard_transformations,
                       local_dict={"x0": _X, "x": _X})
    except Exception:
        return None
    if not isinstance(e, sp.Expr) or not e.free_symbols <= {_X} or not e.free_symbols:
        return None
    return e


def candidates(text: str) -> list:
    """Every formula in x found in free text, longest parsable span first."""
    text = _normalize(text)
    found = []
    for m in _START.finditer(text):
        span = _MATH.match(text, m.end())
        if not span:
            continue
        span = span.group(0)
        for end in range(len(span), 0, -1):
            e = parse(span[:end])
            if e is not None:
                found.append(e)
                break
    return found


def _values(e) -> np.ndarray:
    f = sp.lambdify(_X, e, "numpy")
    with np.errstate(all="ignore"):
        try:
            v = f(_GRID) * np.ones_like(_GRID)
        except Exception:
            return np.full_like(_GRID, np.nan, dtype=complex)
    return np.asarray(v, dtype=complex)


def equivalent(a, b) -> bool:
    va, vb = _values(a), _values(b)
    ok = np.isfinite(va) & np.isfinite(vb) & (np.abs(va.imag) < 1e-9) & (np.abs(vb.imag) < 1e-9)
    if ok.sum() < MIN_POINTS:
        return False
    va, vb = va.real[ok], vb.real[ok]
    close = np.abs(va - vb) <= TOLERANCE * np.maximum(np.abs(vb), 1e-9)
    return close.mean() >= MIN_AGREEMENT


def reported(equation: str, found: list) -> bool:
    """Is the PySR equation among the formulas found in a text?"""
    ref = parse(equation)
    return ref is not None and any(equivalent(c, ref) for c in found)
