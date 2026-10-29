"""Prepares the LaTeX side of the model comparison, once, on the local PC.

For every gold paper that has a LaTeX source (downloaded_latex/<paper>_source.tar.gz):
  gold_latex/<paper>_source.tar.gz            the archive (export-dataset --latex-dir)
  gold_figures_latex/<paper>_source/*.png     its figure files as images, for the
                                              vision pipeline (mode A) and the single
                                              multimodal call (mode B): --figures-dir
  gold_vector_latex/<paper>_source/*.pdf      the vector figures as PDF, for the
                                              library-only mode (C): --vector-dir

Run here and not on Leonardo because EPS figures need Ghostscript to be read at
all (PyMuPDF cannot open EPS: the pipeline's own LaTeX extractor silently loses
them, e.g. all 26 figures of the OTOC paper). Library-only, deterministic, the
same files for every model. Then rsync the three directories to Leonardo.

    python scripts/matrix/prepare_latex_figures.py [--gold-pdfs gold_pdfs] [--latex-dir downloaded_latex]
"""
import argparse
import glob
import os
import shutil
import sys
import tarfile
import tempfile

import pymupdf
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.processing.library_extract import eps_to_pdf  # noqa: E402

# Same longest side as the rendered pages of the single call (Pixtral's limit).
MAX_SIDE_PX = 1540
EXTENSIONS = (".pdf", ".eps", ".png", ".jpg", ".jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-pdfs", default="gold_pdfs")
    ap.add_argument("--latex-dir", default="downloaded_latex")
    args = ap.parse_args()
    if not shutil.which("gs"):
        sys.exit("[!] Ghostscript (gs) not found: EPS figures could not be converted.")

    for d in ("gold_latex", "gold_figures_latex", "gold_vector_latex"):
        os.makedirs(d, exist_ok=True)
    for pdf in sorted(glob.glob(os.path.join(args.gold_pdfs, "*.pdf"))):
        base = os.path.basename(pdf)[:-4]
        tar_path = os.path.join(args.latex_dir, f"{base}_source.tar.gz")
        if not os.path.exists(tar_path):
            print(f"[-] {base[:50]}: no LaTeX source")
            continue
        shutil.copy2(tar_path, "gold_latex")
        png_dir = os.path.join("gold_figures_latex", f"{base}_source")
        vec_dir = os.path.join("gold_vector_latex", f"{base}_source")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(vec_dir, exist_ok=True)
        n_png = n_vec = 0
        with tarfile.open(tar_path, "r:gz") as tar, tempfile.TemporaryDirectory() as tmp:
            for m in tar.getmembers():
                low = m.name.lower()
                if not m.isfile() or not low.endswith(EXTENSIONS):
                    continue
                name = os.path.basename(m.name)
                src = os.path.join(tmp, name)
                with open(src, "wb") as f:
                    f.write(tar.extractfile(m).read())
                stem = os.path.splitext(name)[0]
                if low.endswith((".png", ".jpg", ".jpeg")):
                    # Same longest side as the vector renders: a 10000x2000 PNG
                    # (Transformer-XL) crashed vLLM's multimodal cache with Qwen3.8
                    # and overflowed its context in the single call (job 62440416).
                    with Image.open(src) as im:
                        if max(im.size) > MAX_SIDE_PX:
                            im.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
                        im.save(os.path.join(png_dir, name))
                    n_png += 1
                    continue
                if low.endswith(".eps"):
                    src = eps_to_pdf(src, tmp)
                    if src is None:
                        print(f"[!] {name}: Ghostscript conversion failed")
                        continue
                try:
                    with pymupdf.open(src) as doc:
                        page = doc[0]
                        zoom = min(MAX_SIDE_PX / max(page.rect.width, page.rect.height), 300 / 72)
                        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(os.path.join(png_dir, stem + ".png"))
                except Exception as e:
                    print(f"[!] {name}: not readable ({type(e).__name__}: {e})")
                    continue
                shutil.copy2(src, os.path.join(vec_dir, stem + ".pdf"))
                n_png += 1
                n_vec += 1
        print(f"[+] {base[:50]}: {n_png} figure image(s), {n_vec} vector PDF(s)")


if __name__ == "__main__":
    main()
