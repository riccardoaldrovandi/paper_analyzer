"""Pre-crops the figures of every PDF in --pdf-dir into --out/<paper name>/,
the layout `main.py export-dataset --figures-dir` reads, so every model of a
comparison analyzes exactly the same figures.

Uses the production extraction code (src/processing/image_extractor.py) with
LOCAL_FIGURE_CROP=1, i.e. page filter + plot bounding boxes from the local
vLLM vision server instead of Gemini. Run by crop_figures.sbatch, which sets
the server environment. Resumable: papers whose folder already has figures
are skipped. Writes <out>/manifest.json with what was found per paper.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.processing.image_extractor import LOCAL_FIGURE_CROP, extract_images_from_pdf  # noqa: E402


def kind(path: str) -> str:
    name = os.path.basename(path)
    if "_full_render" in name:
        return "full_page"   # page kept whole: the crop step found no usable box
    if "_crop_" in name:
        return "crop"
    return "raster"          # image embedded in the PDF, kept as is


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not LOCAL_FIGURE_CROP:
        sys.exit("[!] LOCAL_FIGURE_CROP=1 is not set: this would call Gemini, not the local server.")

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    print(f"[+] {len(pdfs)} PDF(s) in {args.pdf_dir} -> {args.out}")
    for i, pdf in enumerate(pdfs, 1):
        name = os.path.splitext(os.path.basename(pdf))[0]
        paper_dir = os.path.join(args.out, name)
        if name in manifest and os.path.isdir(paper_dir) and os.listdir(paper_dir):
            print(f"[~] {i}/{len(pdfs)} {name}: already done, skipped.")
            continue
        print(f"\n========== {i}/{len(pdfs)} {name} ==========")
        images = extract_images_from_pdf(pdf, base_output_dir=args.out)
        counts = {k: sum(kind(p) == k for p in images) for k in ("crop", "full_page", "raster")}
        manifest[name] = {"pdf": pdf, "figures": len(images), **counts}
        print(f"[+] {name}: {len(images)} figure(s) {counts}")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n{'paper':<60} {'crops':>5} {'full':>5} {'raster':>6}")
    for name, m in sorted(manifest.items()):
        print(f"{name[:60]:<60} {m['crop']:>5} {m['full_page']:>5} {m['raster']:>6}")
    print(f"\n[+] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
