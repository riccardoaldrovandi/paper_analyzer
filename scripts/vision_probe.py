"""
Vision-model probe: tells apart a model that cannot SEE a figure from one that
can see it but cannot produce the dense VISION_PROMPT digitization.

For each figure in scripts/probe_figures/ it sends, to the vision server that
is currently running (LOCAL_VISION_SERVER_URL / LOCAL_VISION_MODEL_PATH, same
request format as the pipeline via call_local_vision_server):
  1. the full VISION_PROMPT, then flags series that look synthetic (evenly
     spaced X with piecewise-linear y -- the placeholder pattern LLaVA-NeXT and
     Idefics3 produced in the 2026-10-07 smoke tests);
  2. a few short perception questions, printed next to the answer read by eye
     from the figure.

Run from the repo root while a vision server is up:
    python scripts/vision_probe.py [--out logs/probe_<model>.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.advanced.digitizer import (  # noqa: E402
    VISION_PROMPT, VISION_MODEL_NAME, call_local_vision_server, _parse_vision_output, looks_synthetic,
)
from src.utils.helpers import strip_json_fences  # noqa: E402

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_figures")

# Expected answers were read by eye from the figures (approximate values).
PROBES = {
    "flux_tube.png": [
        ("What are the minimum and maximum values shown on the x axis and on the y axis?",
         "x: 0 to ~5 (ticks 2, 4); y: 0 to 6"),
        ("How many curves are drawn in this plot, and how is each one labeled?",
         "2 curves, labeled '1' (upper) and '2' (lower)"),
        ("What is the y value of the upper curve at x = 2?", "~1.55"),
        ("What value does the upper curve approach at the right end of the plot?", "~0.6"),
    ],
    "gqa_bars.png": [
        ("List every bar label in this chart together with its numerical value.",
         "Mean ~55.55, First ~55.45, Random ~55.23"),
        ("What is the range of the numerical axis?", "~54.3 to ~55.7 (ticks 54.4 ... 55.6)"),
    ],
    "net_proton.png": [
        ("How many panels does this figure have, and what is the y-axis label of the left panel?",
         "2 panels; 'Net-proton High Moments'"),
        ("Is the x axis linear or logarithmic? What is its label?",
         "logarithmic; 'Collision Energy sqrt(s_NN) (GeV)'"),
        ("In the left panel, what is the y value of the red circle at the highest collision energy (200 GeV)?",
         "~0.12"),
        ("List the legend entries of the left panel.",
         "UrQMD 0%-5%, HRG GCE, HRG CE, HRG EV (r=0.5fm)"),
    ],
}


def run_digitization(image_path: str) -> dict:
    t0 = time.time()
    raw = call_local_vision_server(image_path, VISION_PROMPT)
    result = {"seconds": round(time.time() - t0, 1), "raw": raw}
    try:
        parsed = _parse_vision_output(strip_json_fences(raw), VISION_MODEL_NAME)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    if not parsed:
        result["error"] = "no usable series"
        return result
    result["plot_type"] = parsed.get("plot_type")
    if parsed.get("plot_type") == "continuous":
        result["series"] = [{
            "label": s["label"],
            "n_points": len(s["X"]),
            "x_range": [float(s["X"].min()), float(s["X"].max())],
            "y_range": [float(s["y"].min()), float(s["y"].max())],
            "synthetic_suspect": bool(looks_synthetic(s["X"], s["y"])),
        } for s in parsed["series"]]
    else:
        result["points"] = [[str(l), float(x), float(y)] for l, x, y in zip(parsed["labels"], parsed["X"], parsed["y"])]
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe the running local vision server.")
    parser.add_argument("--out", default=f"logs/probe_{VISION_MODEL_NAME}.json")
    args = parser.parse_args()

    print(f"[~] Probing vision model: {VISION_MODEL_NAME}")
    report = {"model": VISION_MODEL_NAME, "figures": {}}
    for fig, questions in PROBES.items():
        path = os.path.join(FIGURE_DIR, fig)
        print(f"\n===== {fig} =====")
        entry = {"digitization": run_digitization(path), "questions": []}
        d = entry["digitization"]
        print(f"[digitization] {d.get('seconds')}s, type={d.get('plot_type')}, error={d.get('error')}")
        for s in d.get("series", []):
            flag = "  <-- SYNTHETIC?" if s["synthetic_suspect"] else ""
            print(f"   curve '{s['label']}': {s['n_points']} pts, x {s['x_range']}, y {s['y_range']}{flag}")
        for p in d.get("points", []):
            print(f"   point {p}")

        for question, expected in questions:
            answer = call_local_vision_server(path, question + " Answer briefly.", max_tokens=200).strip()
            entry["questions"].append({"q": question, "expected": expected, "answer": answer})
            print(f"\n Q: {question}\n   expected: {expected}\n   model:    {answer}")
        report["figures"][fig] = entry

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Full report (incl. raw outputs) saved to {args.out}")


if __name__ == "__main__":
    main()
