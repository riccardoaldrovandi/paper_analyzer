import os
import json
import base64
import threading
import requests
from src.config import OPENROUTER_API_KEY, SYSTEM_PROMPT
from src.utils.retry import QuotaExceededError

# Single source of truth for the free-tier model alias used across the whole
# project (text analysis and, as a Gemini-quota fallback, vision calls too).
FREE_MODEL = "openrouter/free"


def _post_with_hard_timeout(url: str, headers: dict, data: str, timeout: float):
    """
    requests' own `timeout` parameter does NOT cover DNS resolution: urllib3 calls
    socket.getaddrinfo() before it ever applies a timeout to the socket, so if the
    system resolver hangs (e.g. a systemd-resolved hiccup), requests.post() can
    block indefinitely regardless of `timeout` -- no exception is ever raised.
    This runs the call in a background thread and joins it with a hard wall-clock
    timeout, so a resolver hang is bounded no matter what.

    The worker thread is daemon=True and never explicitly killed (Python cannot
    safely kill a thread stuck in a blocking C call): if it times out, the thread
    is simply abandoned and the main flow moves on. That's fine for a short-lived
    CLI run -- a plain (non-daemon) thread via ThreadPoolExecutor would instead
    make the whole process hang at exit waiting to join it.
    """
    box = {}

    def _worker():
        try:
            box["response"] = requests.post(url, headers=headers, data=data, timeout=timeout)
        except Exception as e:
            box["error"] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 10)  # grace period so requests' own timeout can fire first

    if thread.is_alive():
        raise TimeoutError(f"Request to {url} did not complete within {timeout + 10:.0f}s (possibly stuck resolving DNS)")
    if "error" in box:
        raise box["error"]
    return box["response"]

def analyze_with_openrouter(text_content: str) -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        print("[!] Error: OPENROUTER_API_KEY not found in environment variables.")
        return "Unknown OpenRouter Model", "{}"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/riccardoaldro/paper_analyzer",
        "X-Title": "Thesis Paper Analyzer"
    }

    payload = {
        "model": FREE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"--- PAPER CONTENT ---\n{text_content}"}
        ]
    }

    print(f"[~] Sending paper content to OpenRouter ({FREE_MODEL} alias)...")
    try:
        response = _post_with_hard_timeout(url, headers, json.dumps(payload), timeout=90)
        if response.status_code != 200:
            print(f"[!] OpenRouter API Error [HTTP {response.status_code}]: {response.text}")
            return f"{FREE_MODEL} (Failed)", "{}"

        result = response.json()
        model_used = result.get("model", FREE_MODEL)
        content = result['choices'][0]['message']['content']
        return model_used, content
    except Exception as e:
        print(f"[!] Error during OpenRouter analysis: {e}")
        return f"{FREE_MODEL} (Error)", "{}"


def analyze_image_with_openrouter(image_path: str, prompt: str) -> str:
    """
    Sends a single image + text prompt to the same free-tier OpenRouter alias used
    for text analysis. Used as a fallback for the Gemini Vision calls (digitizer,
    smart image filtering/cropping) when Gemini's free-tier quota is exhausted
    (HTTP 429), so the whole project relies on one "free model" convention instead
    of hardcoding a different vision model elsewhere.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment variables.")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/riccardoaldro/paper_analyzer",
        "X-Title": "Thesis Paper Analyzer"
    }
    payload = {
        "model": FREE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ]
    }

    print(f"[~] Sending image to OpenRouter ({FREE_MODEL} alias)...")
    response = _post_with_hard_timeout(url, headers, json.dumps(payload), timeout=90)
    if response.status_code == 429:
        # Distinct from other failures: this specifically means OpenRouter's own
        # free-tier quota is exhausted, not just a bad/unusable response. Since
        # this function is only ever called as the fallback after Gemini has
        # already raised QuotaExceededError, seeing it here means BOTH free
        # providers are genuinely out of quota -- the one condition callers
        # should treat as "stop and checkpoint" rather than "skip and continue".
        raise QuotaExceededError(response.text)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API Error [HTTP {response.status_code}]: {response.text}")

    return response.json()['choices'][0]['message']['content']