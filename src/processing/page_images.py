"""
Images attached to the single multimodal call (export-dataset --mode single):
what a person uploading the paper to a chat model would give it, with no
figure analysis of ours in between.

- PDF: the pages that hold figures or tables, rendered whole (a chat model
  sees the page, not our crops), at most max_images of them.
- LaTeX: the figure files of the source, in the order the .tex includes them.
"""
import os
import re
import glob

import pymupdf

# Longest image side in pixels: Pixtral's limit (Mistral Small 4); Qwen3.8 and
# Gemma 4 resize larger images anyway.
MAX_SIDE_PX = 1540
_CAPTION_RE = re.compile(r"\b(Fig\.|Figure|FIG\.|Table|TABLE)\s*([IVX]+|\d+)")


def _visual_score(page) -> int:
    """How much of a figure/table page this is: captions, images and long drawn paths."""
    long_paths = sum(1 for d in page.get_drawings() if len(d["items"]) >= 5)
    return 3 * len(_CAPTION_RE.findall(page.get_text())) + 2 * len(page.get_images()) + min(long_paths, 20)


def pdf_figure_pages(pdf_path: str, out_dir: str, max_images: int) -> list:
    """PNG renders of the pages with figures or tables (page order), at most max_images, cached in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        scored = [(i, _visual_score(p)) for i, p in enumerate(doc)]
        keep = sorted(i for i, s in sorted(scored, key=lambda t: -t[1])[:max_images] if s > 0)
        paths = []
        for i in keep:
            path = os.path.join(out_dir, f"page{i + 1}.png")
            if not os.path.exists(path):
                page = doc[i]
                zoom = MAX_SIDE_PX / max(page.rect.width, page.rect.height)
                page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(path)
            paths.append(path)
    return paths


def latex_figure_files(tex_text: str, figures_dir: str, max_images: int) -> list:
    """The figure images in figures_dir, ordered as the LaTeX source includes them, at most max_images."""
    files = sorted(p for p in glob.glob(os.path.join(figures_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))
    order = [os.path.splitext(os.path.basename(m))[0] for m in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", tex_text)]
    rank = {name: i for i, name in reversed(list(enumerate(order)))}
    files.sort(key=lambda p: rank.get(os.path.splitext(os.path.basename(p))[0], len(order)))
    return files[:max_images]
