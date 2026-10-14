"""
Test of vLLM guided decoding (schema-constrained JSON) before adopting it in
the pipeline. Every case is run twice, without and with the JSON schema
constraint, under the same seed, to answer three questions:
  1. does it work on vLLM 0.7.3 with our schemas (anyOf, nested arrays)?
  2. how much slower is generation?
  3. does the content get worse (shorter, more generic)?

Modes:
  text    loads the text model given by LOCAL_MODEL_PATH (same settings as
          src/llm/local_client.py) and re-generates the teacher answer for the
          papers in --inputs (user messages of existing export-dataset
          records, so no figures/PySR are redone). Writes two JSONL files in
          export-dataset format (<out>_free.jsonl, <out>_guided.jsonl) to be
          compared locally with `python main.py evaluate`, plus a summary.
  vision  talks to the vision server already running (LOCAL_VISION_SERVER_URL)
          and digitizes every figure under --figures-dir.

Run from the repo root (see scripts/test_guided_json.sbatch).
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm.schemas import ANSWER_SCHEMA, VISION_SCHEMA  # noqa: E402
from src.utils.helpers import strip_json_fences, extract_json_object, repair_json_backslashes  # noqa: E402


def _is_valid_answer(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and all(k in parsed for k in ANSWER_SCHEMA["required"])


def run_text(args):
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from src.config import SYSTEM_PROMPT
    import src.llm.local_client as lc

    # One input per paper, taken from existing export-dataset records.
    inputs = {}
    for path in sorted(glob.glob(args.inputs)):
        for line in open(path, encoding="utf-8"):
            record = json.loads(line)
            inputs.setdefault(record["source_file"], record["messages"][1]["content"])
    if not inputs:
        sys.exit(f"[!] No records found for --inputs '{args.inputs}'.")
    print(f"[~] {len(inputs)} paper(s) from '{args.inputs}', text model {lc.MODEL_NAME}")

    llm = lc._get_llm()
    settings = lc.GENERATION_SETTINGS
    base = dict(
        temperature=settings["temperature"],
        max_tokens=settings["max_new_tokens"],
        repetition_penalty=settings["repetition_penalty"],
        frequency_penalty=settings["frequency_penalty"],
        seed=settings["seed"],
    )
    modes = {
        "free": SamplingParams(**base),
        "guided": SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=ANSWER_SCHEMA)),
    }

    summary = []
    files = {mode: open(f"{args.out}_{mode}.jsonl", "w", encoding="utf-8") for mode in modes}
    for source, user_text in inputs.items():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{lc.USER_PREFIX}{user_text}"},
        ]
        for mode, params in modes.items():
            print(f"\n[~] {os.path.basename(source)[:50]} | {mode}")
            t0 = time.time()
            try:
                output = llm.chat(messages, params)[0].outputs[0]
                raw, n_tokens, finish = output.text, len(output.token_ids), output.finish_reason
                error = None
            except Exception as e:  # e.g. schema not supported by the backend
                raw, n_tokens, finish, error = "", 0, None, f"{type(e).__name__}: {e}"
            seconds = round(time.time() - t0, 1)
            # Same cleanup as analyze_with_local, so "free" is exactly production.
            cleaned = repair_json_backslashes(extract_json_object(strip_json_fences(raw)))
            valid = _is_valid_answer(cleaned)
            row = {"source_file": source, "mode": mode, "seconds": seconds, "output_tokens": n_tokens,
                   "finish_reason": finish, "valid_json": valid, "answer_chars": len(cleaned), "error": error}
            summary.append(row)
            print(f"    {seconds}s, {n_tokens} tokens, finish={finish}, valid JSON={valid}" + (f", ERROR {error}" if error else ""))
            if valid:
                files[mode].write(json.dumps({
                    "source_file": source,
                    "provider": f"{lc.MODEL_NAME} [{mode}]",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": cleaned},
                    ],
                }, ensure_ascii=False) + "\n")
            else:
                print(f"    raw start: {raw[:200]!r}")
    for f in files.values():
        f.close()

    with open(f"{args.out}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== TEXT SUMMARY =====")
    for mode in modes:
        rows = [r for r in summary if r["mode"] == mode]
        ok = sum(r["valid_json"] for r in rows)
        avg = sum(r["seconds"] for r in rows) / len(rows)
        print(f"{mode:<7} valid {ok}/{len(rows)}, avg {avg:.1f}s, avg {sum(r['output_tokens'] for r in rows) / len(rows):.0f} output tokens")
    print(f"[+] Records: {args.out}_free.jsonl / {args.out}_guided.jsonl, summary: {args.out}_summary.json")


def run_vision(args):
    from src.advanced.digitizer import (
        VISION_PROMPT, VISION_MODEL_NAME, call_local_vision_server, _parse_vision_output, looks_synthetic,
    )

    images = sorted(p for p in glob.glob(os.path.join(args.figures_dir, "**", "*"), recursive=True)
                    if p.lower().endswith((".png", ".jpg", ".jpeg")))
    if not images:
        sys.exit(f"[!] No images under '{args.figures_dir}'.")
    print(f"[~] {len(images)} figure(s), vision model {VISION_MODEL_NAME}")

    modes = {"free": None, "guided": {"guided_json": VISION_SCHEMA}}
    summary = []
    for image in images:
        for mode, extra in modes.items():
            print(f"\n[~] {os.path.basename(image)[-45:]} | {mode}")
            t0 = time.time()
            row = {"figure": image, "mode": mode}
            try:
                raw = call_local_vision_server(image, VISION_PROMPT, max_tokens=4096, extra_body=extra)
                row.update(seconds=round(time.time() - t0, 1), raw_chars=len(raw), raw=raw)
                parsed = _parse_vision_output(strip_json_fences(raw), VISION_MODEL_NAME)
                if parsed is None:
                    row.update(parsed=False)
                elif parsed.get("plot_type") == "continuous":
                    row.update(parsed=True, plot_type="continuous", n_series=len(parsed["series"]),
                               synthetic_series=sum(bool(looks_synthetic(s["X"], s["y"])) for s in parsed["series"]))
                else:
                    row.update(parsed=True, plot_type="discrete", n_points=len(parsed["labels"]))
            except Exception as e:
                row.update(seconds=round(time.time() - t0, 1), parsed=False, error=f"{type(e).__name__}: {e}")
            summary.append(row)
            print("    " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("figure", "mode", "raw")))

    with open(f"{args.out}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== VISION SUMMARY =====")
    for mode in modes:
        rows = [r for r in summary if r["mode"] == mode]
        ok = sum(bool(r.get("parsed")) for r in rows)
        avg = sum(r["seconds"] for r in rows) / len(rows)
        synth = sum(r.get("synthetic_series", 0) for r in rows)
        print(f"{mode:<7} parsed {ok}/{len(rows)}, avg {avg:.1f}s, synthetic series {synth}")
    print(f"[+] Summary (with raw outputs): {args.out}_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Compare free vs schema-guided JSON generation.")
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("text")
    t.add_argument("--inputs", required=True, help="Glob of export-dataset JSONL files whose user messages are reused")
    t.add_argument("--out", required=True, help="Output path prefix")
    v = sub.add_parser("vision")
    v.add_argument("--figures-dir", default="comparison_figures")
    v.add_argument("--out", required=True, help="Output path prefix")
    args = parser.parse_args()
    run_text(args) if args.mode == "text" else run_vision(args)


if __name__ == "__main__":
    main()
