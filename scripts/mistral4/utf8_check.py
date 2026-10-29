"""Does the Mistral Small 4 server corrupt non-ASCII characters, and with which settings?

In the 2026-10-27 matrix Mistral's answers contained U+FFFD ("ρ��", "��α"): a
multi-byte UTF-8 character split across byte-fallback tokens and decoded
piecewise. The answers were generated with skip_special_tokens=false (needed
so the reasoning parser sees [THINK]). This asks the running server to repeat
a line of physics symbols with each setting and counts the broken characters.

    python scripts/mistral4/utf8_check.py   (inside a job, server already up: LOCAL_TEXT_SERVER_URL)
"""
import os
import json

import requests

URL = os.environ.get("LOCAL_TEXT_SERVER_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("LOCAL_MODEL_PATH", "")
TEXT = "ρ∞ ≠ 0, α̃ = β̃ = −3, φ(x) → 0, Φ_E = 2π ∫₀^∞ f v ρ dρ, ħ ↔ μ l³, ε^{−1/2}, Δp_t^(B), √s_NN = 7.7 GeV, κσ², Θ(4, 4), λ/t, Ω₀α/h"
SETTINGS = {
    "skip_special_tokens=false, reasoning none (matrix setting)": {"skip_special_tokens": False, "reasoning_effort": "none"},
    "skip_special_tokens=true, reasoning none": {"skip_special_tokens": True, "reasoning_effort": "none"},
    "skip_special_tokens=false, reasoning high (matrix final call)": {"skip_special_tokens": False, "reasoning_effort": "high"},
    "skip_special_tokens=true, reasoning high": {"skip_special_tokens": True, "reasoning_effort": "high"},
}


def main():
    for name, extra in SETTINGS.items():
        payload = {"model": MODEL, "temperature": 0, "max_tokens": 2048, **extra,
                   "messages": [{"role": "user", "content": f"Repeat exactly this line, nothing else:\n{TEXT}"}]}
        try:
            r = requests.post(f"{URL}/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        except Exception as e:
            print(f"[!] {name}: request failed ({type(e).__name__}: {e})")
            continue
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        print(f"[{'OK' if content.count(chr(0xFFFD)) == 0 else 'BROKEN'}] {name}: "
              f"{content.count(chr(0xFFFD))} U+FFFD in the answer, {reasoning.count(chr(0xFFFD))} in the reasoning, "
              f"reasoning split: {bool(reasoning)}, [THINK] left in the answer: {'[THINK]' in content}")
        print(f"      {json.dumps(content.strip()[:200], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
