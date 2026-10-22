"""
Step 1b (GPU, Leonardo, inside run_atomic.sbatch): splits the sentence-level
claims of claims.jsonl into atomic claims with a vLLM server (Qwen3.8, thinking
off). Talks to the server over HTTP with the standard library only.

Writes (a) one line per sentence with the raw and checked atoms (resumable),
(b) atomic_claims.jsonl, same format as claims.jsonl, so judge_claims.py and
report.py work on it unchanged. A sentence whose atoms fail the checks of
src/evaluation/atomic.py is kept whole.

    python scripts/groundedness/decompose_claims.py --claims logs/groundedness/claims.jsonl \
        --url http://localhost:20000/v1 --atoms logs/groundedness/atoms.jsonl \
        --out logs/groundedness/atomic_claims.jsonl --runs qwen38_gold_think_s0 ...
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.evaluation.atomic import SCHEMA, build_messages, check_atoms  # noqa: E402


def post(url: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def decompose(url: str, model: str, sentence: str, context: str, retries: int = 3):
    """The model's list of atoms, or None if every attempt failed."""
    body = {"model": model, "messages": build_messages(sentence, context), "temperature": 0, "max_tokens": 1024,
            "response_format": {"type": "json_schema", "json_schema": {"name": "atoms", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(retries):
        try:
            content = post(f"{url}/chat/completions", body)["choices"][0]["message"]["content"]
            return json.loads(content)["claims"]
        except Exception as e:  # noqa: BLE001 - network, HTTP and JSON errors all mean "try again"
            print(f"[!] attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:20000/v1")
    ap.add_argument("--atoms", required=True, help="per-sentence results (resumable)")
    ap.add_argument("--out", required=True, help="atomic claims, same format as claims.jsonl")
    ap.add_argument("--runs", nargs="+", help="only these runs (default: all in --claims)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only the first N sentences (test)")
    args = ap.parse_args()

    with open(args.claims, encoding="utf-8") as f:
        sentences = [json.loads(line) for line in f if line.strip()]
    if args.runs:
        sentences = [s for s in sentences if s["run"] in set(args.runs)]
    if args.limit:
        sentences = sentences[:args.limit]

    # Surrounding text of a sentence = all sentences of the same field of the same answer.
    field_text = defaultdict(list)
    for s in sentences:
        field_text[(s["run"], s["source_file"], s["field"])].append(s["claim"])

    done = {}
    if os.path.exists(args.atoms):
        with open(args.atoms, encoding="utf-8") as f:
            done = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    todo = [s for s in sentences if s["id"] not in done]
    print(f"[~] {len(sentences)} sentences, {len(done)} already done, {len(todo)} to do")

    if todo:
        model = json.loads(urllib.request.urlopen(f"{args.url}/models", timeout=30).read())["data"][0]["id"]
        print(f"[~] Server model: {model}")

        def work(s):
            context = " ".join(field_text[(s["run"], s["source_file"], s["field"])])
            raw = decompose(args.url, model, s["claim"], context)
            atoms, fallback = (([s["claim"]], "request_failed") if raw is None
                               else check_atoms(s["claim"], context, raw))
            return {"id": s["id"], "raw": raw, "atoms": atoms, "fallback": fallback}

        t0 = time.time()
        with open(args.atoms, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
            for i, res in enumerate(pool.map(work, todo), 1):
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
                done[res["id"]] = res
                if i % 50 == 0 or i == len(todo):
                    print(f"[~] {i}/{len(todo)} sentences, {time.time() - t0:.0f}s", flush=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for s in sentences:
            for k, atom in enumerate(done[s["id"]]["atoms"]):
                out.write(json.dumps({**{x: s[x] for x in ("run", "provider", "source_file", "field")},
                                      "id": f"{s['id']}#{k}", "sentence_id": s["id"], "claim": atom,
                                      "fallback": done[s["id"]]["fallback"]}, ensure_ascii=False) + "\n")
                n += 1
    reasons = Counter(done[s["id"]]["fallback"] for s in sentences)
    print(f"[+] {n} atomic claims from {len(sentences)} sentences ({n / len(sentences):.2f} per sentence) -> {args.out}")
    print(f"[~] Kept whole (fallback): {sum(v for k, v in reasons.items() if k)}/{len(sentences)} "
          f"{ {k: v for k, v in reasons.items() if k} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
