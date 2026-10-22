"""Smoke test for a running Qwen3.8-27B vLLM server (started by smoke_test.sbatch).

Four checks, each one a capability the pipeline relies on:
  1. text, thinking OFF  -> plain JSON answer (how the pipeline will call it)
  2. text, thinking ON   -> reasoning separated from the answer by --reasoning-parser
  3. image, thinking OFF -> reads a real paper figure
  4. guided JSON         -> response_format json_schema is honored

Standard library only, so it runs unchanged in the venv and in the container.
Exit code 0 only if every check passes; details go to --out as JSON.
"""
import argparse
import base64
import json
import sys
import time
import urllib.request

MODEL = None  # set in main() from the server's /v1/models (the model's path)
# Model card's recommended sampling for non-thinking (instruct) mode.
NO_THINK = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": False},
}
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def chat(base_url: str, messages: list, max_tokens: int, **extra) -> tuple[dict, float]:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, **extra}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp), time.time() - start


def reasoning_of(msg: dict) -> str:
    # Newer vLLM releases name the field "reasoning", older ones "reasoning_content".
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json")
    return json.loads(text)


def check_text_no_think(url):
    prompt = ('Return ONLY a JSON object {"law": str, "formula": str} for the '
              "relation between the energy and the momentum of a free relativistic particle.")
    r, dt = chat(url, [{"role": "user", "content": prompt}], 512, **NO_THINK)
    msg = r["choices"][0]["message"]
    data = parse_json(msg["content"])
    assert {"law", "formula"} <= data.keys(), f"missing keys: {data}"
    assert not reasoning_of(msg), "thinking was not disabled"
    return r, dt, data


def check_text_think(url):
    r, dt = chat(url, [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], 4096, **THINK)
    msg = r["choices"][0]["message"]
    assert reasoning_of(msg), "no reasoning in the reply: reasoning parser not active?"
    assert "391" in (msg.get("content") or ""), f"wrong/empty answer: {msg.get('content')!r}"
    return r, dt, {"answer": msg["content"], "reasoning_chars": len(reasoning_of(msg))}


def check_image(url, figure):
    with open(figure, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = ('This is a figure from a physics paper. Return ONLY a JSON object '
              '{"chart_type": str, "x_axis": str, "y_axis": str, "description": str}.')
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]
    r, dt = chat(url, messages, 1024, **NO_THINK)
    data = parse_json(r["choices"][0]["message"]["content"])
    assert {"chart_type", "x_axis", "y_axis"} <= data.keys(), f"missing keys: {data}"
    return r, dt, data


def check_guided_json(url):
    schema = {"type": "object", "additionalProperties": False, "required": ["title", "year", "keywords"],
              "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                             "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 5}}}
    text = ("The anti-k_t jet clustering algorithm, by Cacciari, Salam and Soyez, published in 2008, "
            "introduces a sequential recombination jet algorithm for hadron colliders.")
    r, dt = chat(url, [{"role": "user", "content": f"Extract metadata from: {text}"}], 512,
                 response_format={"type": "json_schema", "json_schema": {"name": "meta", "schema": schema}},
                 **NO_THINK)
    data = json.loads(r["choices"][0]["message"]["content"])
    assert data["year"] == 2008, f"year: {data['year']}"
    return r, dt, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--figure", required=True)
    ap.add_argument("--out", required=True)
    # Other models (scripts/gemma4): their own request settings, as JSON merged into
    # the request body, e.g. '{"chat_template_kwargs": {"enable_thinking": true}}'.
    ap.add_argument("--nothink-json", help="replaces the default non-thinking request settings")
    ap.add_argument("--think-json", help="replaces the default thinking request settings")
    args = ap.parse_args()
    global NO_THINK, THINK
    if args.nothink_json:
        NO_THINK = json.loads(args.nothink_json)
    if args.think_json:
        THINK = json.loads(args.think_json)
    url = f"http://localhost:{args.port}/v1"
    global MODEL
    with urllib.request.urlopen(f"{url}/models", timeout=30) as resp:
        MODEL = json.load(resp)["data"][0]["id"]

    checks = [("text_no_think", lambda: check_text_no_think(url)),
              ("text_think", lambda: check_text_think(url)),
              ("image", lambda: check_image(url, args.figure)),
              ("guided_json", lambda: check_guided_json(url))]
    results = {}
    for name, fn in checks:
        try:
            r, dt, data = fn()
            out_tok = r["usage"]["completion_tokens"]
            results[name] = {"ok": True, "seconds": round(dt, 1), "prompt_tokens": r["usage"]["prompt_tokens"],
                             "completion_tokens": out_tok, "tok_per_s": round(out_tok / dt, 1), "result": data}
            print(f"[+] {name}: OK in {dt:.1f}s, {out_tok} tokens ({out_tok / dt:.1f} tok/s) -> {json.dumps(data)[:200]}")
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[!] {name}: FAILED -- {type(e).__name__}: {e}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    passed = sum(v["ok"] for v in results.values())
    print(f"\n[~] {passed}/{len(checks)} checks passed. Details: {args.out}")
    sys.exit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
