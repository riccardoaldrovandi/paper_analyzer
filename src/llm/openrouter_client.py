import os
import json
import base64
import requests
from src.config import OPENROUTER_API_KEY, SYSTEM_PROMPT

# Single source of truth for the free-tier model alias used across the whole
# project (text analysis and, as a Gemini-quota fallback, vision calls too).
FREE_MODEL = "openrouter/free"

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
        response = requests.post(url, headers=headers, data=json.dumps(payload))
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
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API Error [HTTP {response.status_code}]: {response.text}")

    return response.json()['choices'][0]['message']['content']