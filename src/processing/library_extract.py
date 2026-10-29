"""
Figure and table data read with libraries only, no model call: mode "libs" of
the model comparison (main.py export-dataset --mode libs).

The pipeline's figure analysis asks a vision model to read points off a
picture. Most physics papers, though, draw their plots as vector graphics, so
the exact coordinates are already in the file:

- PDF pages and figure PDFs: every curve is a stroked path. PyMuPDF returns
  those paths in page coordinates, and the numeric tick labels around each
  plot give the page -> data mapping (linear or log10, fitted on the labels).
- LaTeX: pgfplots figures state their data in the source
  (\\addplot coordinates {...}); figure files (.pdf, .eps converted with
  Ghostscript) go through the same vector reader as PDF pages.
- Tables: PyMuPDF's find_tables() on the PDF pages that mention a table; a
  LaTeX source already contains its tabular environments as text.

The result has the plot_data format of digitize_plot(), so the same PySR step
and ADVANCED PLOT ANALYSIS block are used for every mode. Raster figures,
bar charts and axes without numeric tick labels are skipped and reported,
never guessed: this mode measures what libraries alone can recover.
"""
import os
import re
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
import pymupdf

# Limits that keep one paper's PySR time bounded (each curve is one fit).
MAX_PANELS_PER_PAPER = 12
MAX_CURVES_PER_PANEL = 6
MAX_POINTS_PER_CURVE = 200
MIN_CURVE_POINTS = 5
# Paths whose bounding box is at most this size (pt) are plot markers.
MARKER_SIZE = 8.0
# Pages with more drawing commands than this are skipped by find_tables(),
# which gets very slow on them (dense plots, not tables).
TABLE_SCAN_MAX_DRAWINGS = 3000

_NUMBER_RE = re.compile(r"^[−\-–]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][−\-–+]?\d+)?$")


def _number(text: str):
    t = text.strip().rstrip(",;")
    if not _NUMBER_RE.match(t):
        return None
    return float(t.replace("−", "-").replace("–", "-"))


# ------------------------------------------------------------ vector plots --

def _path_points(path: dict):
    """The points of a stroked path, or None for rectangles/quads (frames, bars, fills)."""
    pts = []
    for item in path["items"]:
        if item[0] == "l":
            pts += [item[1], item[2]]
        elif item[0] == "c":
            pts += [item[1], item[4]]
        else:
            return None
    out = []
    for p in pts:
        if not out or abs(p.x - out[-1][0]) > 1e-3 or abs(p.y - out[-1][1]) > 1e-3:
            out.append((p.x, p.y))
    return out


def _is_axis_aligned(points: list) -> bool:
    xs = {round(x, 1) for x, _ in points}
    ys = {round(y, 1) for _, y in points}
    return len(xs) <= 2 or len(ys) <= 2 and len(xs) <= 4


def _style(path: dict) -> tuple:
    color = tuple(round(c, 3) for c in (path.get("color") or ()))
    return color, round(path.get("width") or 0, 2), path.get("dashes") or ""


def _hex(color: tuple) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in color[:3]) if color else "none"


def _candidate_series(drawings: list) -> list:
    """Curves (long stroked polylines) and scatter markers, grouped by style."""
    curves, markers = {}, {}
    for d in drawings:
        if not d.get("color"):  # fill only: bands, bars, shapes
            continue
        pts = _path_points(d)
        if pts is None:
            continue
        r = d["rect"]
        style = _style(d)
        if r.width <= MARKER_SIZE and r.height <= MARKER_SIZE and len(pts) >= 3:
            markers.setdefault(style, []).append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        elif len(pts) >= MIN_CURVE_POINTS and not _is_axis_aligned(pts):
            curves.setdefault(style, []).append(pts)
    series = [{"style": s, "points": [p for part in parts for p in part], "kind": "curve"} for s, parts in curves.items()]
    series += [{"style": s, "points": pts, "kind": "markers"} for s, pts in markers.items() if len(pts) >= MIN_CURVE_POINTS]
    return series


def _bbox(points: list) -> tuple:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _cluster(series: list, gap: float = 15.0) -> list:
    """Groups series whose bounding boxes overlap (one group = one plot panel)."""
    boxes = [_bbox(s["points"]) for s in series]
    groups = []
    for i, b in enumerate(boxes):
        hits = [g for g in groups if any(
            b[0] - gap <= boxes[j][2] and boxes[j][0] - gap <= b[2] and b[1] - gap <= boxes[j][3] and boxes[j][1] - gap <= b[3]
            for j in g)]
        merged = [i] + [j for g in hits for j in g]
        groups = [g for g in groups if g not in hits] + [merged]
    return [[series[j] for j in sorted(g)] for g in groups]


def _frames(drawings: list, page_rect) -> list:
    """
    Plot frames: dark stroked rectangles, drawn either as one rectangle (ROOT,
    Mathematica) or as four separate sides (matplotlib spines). Only the
    outermost ones, so legend boxes inside a panel do not count.
    """
    rects, hs, vs = [], [], []
    for d in drawings:
        color = d.get("color")
        if not color or sum(color[:3]) > 1.5:
            continue
        r = d["rect"]
        items = d["items"]
        if len(items) == 1 and items[0][0] == "re" and r.width >= 60 and r.height >= 30:
            rects.append(tuple(r))
        elif len(items) == 1 and items[0][0] == "l":
            if r.height < 1 and r.width >= 40:
                hs.append(r)
            elif r.width < 1 and r.height >= 30:
                vs.append(r)
    near = lambda a, b: abs(a - b) <= 2
    for top in hs:
        for bottom in hs:
            if bottom.y0 - top.y0 < 30 or not (near(top.x0, bottom.x0) and near(top.x1, bottom.x1)):
                continue
            if any(near(v.x0, top.x0) and near(v.y0, top.y0) and near(v.y1, bottom.y0) for v in vs):
                rects.append((top.x0, top.y0, top.x1, bottom.y0))
    page_area = page_rect.width * page_rect.height
    rects = {tuple(round(v) for v in r) for r in rects
             if r[0] >= page_rect.x0 - 1 and r[1] >= page_rect.y0 - 1 and r[2] <= page_rect.x1 + 1 and r[3] <= page_rect.y1 + 1
             and (r[2] - r[0]) * (r[3] - r[1]) < 0.9 * page_area}
    inside = lambda a, b: a != b and a[0] >= b[0] - 2 and a[1] >= b[1] - 2 and a[2] <= b[2] + 2 and a[3] <= b[3] + 2
    return sorted(r for r in rects if not any(inside(r, o) for o in rects))


def _split_by_frames(series: list, frames: list) -> list:
    """One group of series per frame, each series cut to the points inside that frame."""
    groups = []
    for f in frames:
        group = []
        for s in series:
            pts = [p for p in s["points"] if f[0] - 1 <= p[0] <= f[2] + 1 and f[1] - 1 <= p[1] <= f[3] + 1]
            if len(pts) >= MIN_CURVE_POINTS:
                group.append(dict(s, points=pts))
        if group:
            groups.append((f, group))
    return groups


def _fit_axis(pixels: list, values: list):
    """(transform, 'linear'|'log') mapping a page coordinate to a data value, or None."""
    p, v = np.array(pixels, float), np.array(values, float)
    if len(p) < 3 or np.ptp(p) < 10 or np.ptp(v) == 0:
        return None
    for scale in ("linear", "log"):
        target = v if scale == "linear" else (np.log10(v) if (v > 0).all() else None)
        if target is None:
            continue
        a, b = np.polyfit(p, target, 1)
        if a == 0:
            continue
        resid = np.abs(a * p + b - target).max()
        if resid <= 0.01 * np.ptp(target):
            if scale == "linear":
                return (lambda x, a=a, b=b: a * x + b), scale
            return (lambda x, a=a, b=b: 10 ** (a * x + b)), scale
    return None


def _axis_labels(words: list, box: tuple, axis: str) -> list:
    """Numeric tick labels next to a panel: one row below it (x) or one column left of it (y)."""
    x0, y0, x1, y1 = box
    nums = []
    for w in words:
        value = _number(w[4])
        if value is None:
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        nums.append((cx, cy, w[2], value))
    if axis == "x":
        cands = [n for n in nums if y1 - 5 <= n[1] <= y1 + 60 and x0 - 30 <= n[0] <= x1 + 30]
        key = lambda n: round(n[1] / 3)
    else:
        cands = [n for n in nums if x0 - 80 <= n[2] <= x0 + 5 and y0 - 30 <= n[1] <= y1 + 30]
        key = lambda n: round(n[2] / 3)
    rows = {}
    for n in cands:
        rows.setdefault(key(n), []).append(n)
    best = [r for r in rows.values() if len(r) >= 3]
    if not best:
        return []
    # The row/column nearest to the panel: tick labels, not a second plot's.
    if axis == "x":
        row = min(best, key=lambda r: abs(np.mean([n[1] for n in r]) - y1))
        return [(n[0], n[3]) for n in row]
    col = min(best, key=lambda r: abs(np.mean([n[2] for n in r]) - x0))
    return [(n[1], n[3]) for n in col]


def _legend_label(series: dict, drawings: list, words: list):
    """Text right of a short legend line drawn in the same style as the series."""
    for d in drawings:
        if _style(d) != series["style"]:
            continue
        pts = _path_points(d)
        r = d["rect"]
        if not pts or len(pts) > 3 or not (8 <= r.width <= 60) or r.height > 2:
            continue
        cy = (r.y0 + r.y1) / 2
        line = sorted((w for w in words if abs((w[1] + w[3]) / 2 - cy) < 4 and r.x1 < w[0] < r.x1 + 200), key=lambda w: w[0])
        text, last = [], r.x1
        for w in line:
            if w[0] - last > 12:
                break
            text.append(w[4])
            last = w[2]
        if text:
            return " ".join(text)
    return None


def _to_xy(points: list, fx, fy) -> tuple:
    pts = sorted(points)
    X = np.array([fx(x) for x, _ in pts])
    y = np.array([fy(yy) for _, yy in pts])
    if len(X) > MAX_POINTS_PER_CURVE:
        idx = np.linspace(0, len(X) - 1, MAX_POINTS_PER_CURVE).round().astype(int)
        X, y = X[idx], y[idx]
    return [float(v) for v in X], [float(v) for v in y]


def vector_plots(page, name: str) -> tuple:
    """
    ([(panel name, plot_data)], [skip notes]) for the vector plots drawn on one
    PDF page. plot_data matches digitize_plot(): {"plot_type": "continuous",
    "series": [{"label", "X", "y"}]}.
    """
    drawings = page.get_drawings()
    series = _candidate_series(drawings)
    if not series:
        return [], []
    words = page.get_text("words")
    found, notes = [], []
    frames = _frames(drawings, page.rect)
    if frames:
        panels = _split_by_frames(series, frames)
    else:
        panels = [(_bbox([p for s in g for p in s["points"]]), g) for g in _cluster(series)]
    for k, (box, group) in enumerate(panels, 1):
        if box[2] - box[0] < 40 or box[3] - box[1] < 20:
            continue  # a glyph or a legend sample, not a plot
        panel = f"{name}, vector plot {k}"
        xl, yl = _axis_labels(words, box, "x"), _axis_labels(words, box, "y")
        if len(xl) < 3:
            # Stacked panels share the x axis: its labels sit under the lowest
            # frame of the same column.
            below = [f for f, _ in panels if f[1] >= box[3] - 2 and abs(f[0] - box[0]) <= 3 and abs(f[2] - box[2]) <= 3]
            if below:
                xl = _axis_labels(words, max(below, key=lambda f: f[3]), "x")
        fx = _fit_axis([p for p, _ in xl], [v for _, v in xl])
        fy = _fit_axis([p for p, _ in yl], [v for _, v in yl])
        if fx is None or fy is None:
            notes.append(f"{panel}: {len(group)} curve(s) found, but no numeric "
                         f"{'x' if fx is None else 'y'}-axis labels to calibrate them (not digitized)")
            continue
        out = []
        for i, s in enumerate(sorted(group, key=lambda s: -len(s["points"]))[:MAX_CURVES_PER_PANEL], 1):
            X, y = _to_xy(s["points"], fx[0], fy[0])
            if np.ptp(X) == 0:
                continue
            label = _legend_label(s, drawings, words) or f"{s['kind']} {i} ({_hex(s['style'][0])})"
            out.append({"label": label, "X": X, "y": y})
        if out:
            axes = f"x {fx[1]}, y {fy[1]}"
            found.append((f"{panel} ({axes} axes)", {"plot_type": "continuous", "series": out}))
    return found, notes


# -------------------------------------------------------------------- PDF --

_TABLE_RE = re.compile(r"\b(Table|TABLE|Tab\.)\s*([IVX]+|\d+)")


def pdf_tables_markdown(pdf_path: str) -> list:
    """Markdown of every table PyMuPDF finds on the pages that mention a table."""
    out = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            if not _TABLE_RE.search(page.get_text()):
                continue
            if sum(len(d["items"]) for d in page.get_drawings()) > TABLE_SCAN_MAX_DRAWINGS:
                continue
            try:
                tables = page.find_tables().tables
            except Exception:
                continue
            for k, t in enumerate(tables, 1):
                md = t.to_markdown().strip()
                if md.count("\n") >= 2:
                    out.append(f"[Table found on page {i}, #{k}]\n{md}")
    return out


def pdf_plots(pdf_path: str) -> tuple:
    found, notes = [], []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            f, n = vector_plots(page, f"page {i}")
            found += f
            notes += n
            if not f and page.get_images():
                notes.append(f"page {i}: raster figure(s), not readable without a model")
    return found, notes


# ------------------------------------------------------------------ LaTeX --

def eps_to_pdf(eps_path: str, out_dir: str):
    """Converts an EPS figure to PDF with Ghostscript (PyMuPDF cannot open EPS). None if gs is missing or fails."""
    gs = shutil.which("gs")
    if not gs:
        return None
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(eps_path))[0] + ".pdf")
    try:
        subprocess.run([gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop", "-sDEVICE=pdfwrite",
                        f"-sOutputFile={out}", eps_path], check=True, timeout=120, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if os.path.exists(out) else None


_AXIS_RE = re.compile(r"\\begin\{(axis|semilogxaxis|semilogyaxis|loglogaxis)\}(\[.*?\])?(.*?)\\end\{\1\}", re.S)
_ADDPLOT_RE = re.compile(r"\\addplot3?\+?\s*(?:\[[^\]]*\])?\s*coordinates\s*\{(.*?)\}", re.S)
_COORD_RE = re.compile(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)")
_LEGEND_RE = re.compile(r"\\addlegendentry\s*\{([^}]*)\}")
_LEGEND_OPT_RE = re.compile(r"legend entries\s*=\s*\{([^}]*)\}")


def pgfplots_data(tex: str, source_name: str) -> list:
    """[(name, plot_data)] for every pgfplots axis whose \\addplot coordinates are numeric."""
    found = []
    for k, m in enumerate(_AXIS_RE.finditer(tex), 1):
        opts, body = m.group(2) or "", m.group(3)
        legends = _LEGEND_RE.findall(body)
        if not legends and _LEGEND_OPT_RE.search(opts):
            legends = [e.strip() for e in _LEGEND_OPT_RE.search(opts).group(1).split(",")]
        series = []
        for i, plot in enumerate(_ADDPLOT_RE.finditer(body)):
            pairs = [(_number(a), _number(b)) for a, b in _COORD_RE.findall(plot.group(1))]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) < 2:
                continue  # symbolic coordinates (bar charts) are not curves
            pairs.sort()
            label = legends[i] if i < len(legends) else f"plot {i + 1}"
            series.append({"label": re.sub(r"[{}$\\]", "", label).strip() or f"plot {i + 1}",
                           "X": [a for a, _ in pairs], "y": [b for _, b in pairs]})
        if series:
            found.append((f"{source_name}, pgfplots axis {k} ({m.group(1)})", {"plot_type": "continuous", "series": series}))
    return found


def latex_plots(tar_path: str, vector_dir: str = None) -> tuple:
    """
    Plots of a LaTeX source: pgfplots data from the .tex files, plus the vector
    figure files. vector_dir, if set, holds those files already as PDFs
    (scripts/matrix/prepare_latex_figures.py, which converts EPS where gs exists);
    otherwise they are read from the archive and EPS are converted here.
    """
    found, notes = [], []
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".tex"):
                tex = tar.extractfile(m).read().decode("utf-8", "ignore")
                found += pgfplots_data(re.sub(r"(?<!\\)%.*", "", tex), os.path.basename(m.name))
        with tempfile.TemporaryDirectory() as tmp:
            if vector_dir and os.path.isdir(vector_dir):
                files = sorted(os.path.join(vector_dir, f) for f in os.listdir(vector_dir) if f.lower().endswith(".pdf"))
            else:
                files = []
                for m in tar.getmembers():
                    low = m.name.lower()
                    if not m.isfile() or not low.endswith((".pdf", ".eps")):
                        continue
                    path = os.path.join(tmp, os.path.basename(m.name))
                    with open(path, "wb") as f:
                        f.write(tar.extractfile(m).read())
                    if low.endswith(".eps"):
                        path = eps_to_pdf(path, tmp)
                        if path is None:
                            notes.append(f"{os.path.basename(m.name)}: EPS figure, Ghostscript unavailable (not read)")
                            continue
                    files.append(path)
            for path in files:
                try:
                    with pymupdf.open(path) as doc:
                        f, n = vector_plots(doc[0], os.path.basename(path))
                except Exception as e:
                    notes.append(f"{os.path.basename(path)}: unreadable ({type(e).__name__})")
                    continue
                found += f
                notes += n
                if not f and not n:
                    notes.append(f"{os.path.basename(path)}: no vector curves (raster or diagram)")
    return found, notes


def extract(pdf: str = None, latex: str = None, vector_dir: str = None) -> tuple:
    """(plots, tables markdown, notes) of one paper, capped at MAX_PANELS_PER_PAPER plots."""
    if pdf:
        plots, notes = pdf_plots(pdf)
        tables = pdf_tables_markdown(pdf)
    else:
        plots, notes = latex_plots(latex, vector_dir)
        tables = []
    # Same types as digitize_plot()'s output: PySR needs numpy arrays.
    for _, plot_data in plots:
        for series in plot_data["series"]:
            series["X"], series["y"] = np.asarray(series["X"], float), np.asarray(series["y"], float)
    if len(plots) > MAX_PANELS_PER_PAPER:
        notes.append(f"{len(plots) - MAX_PANELS_PER_PAPER} more plot panel(s) not analyzed (cap {MAX_PANELS_PER_PAPER})")
        # Panels with more data points first: the main results, not insets.
        plots = sorted(plots, key=lambda p: -sum(len(s["X"]) for s in p[1]["series"]))[:MAX_PANELS_PER_PAPER]
    return plots, tables, notes
