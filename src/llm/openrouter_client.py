import os
import json
import requests
from src.config import OPENROUTER_API_KEY, SYSTEM_PROMPT

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
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"--- PAPER CONTENT ---\n{text_content}"}
        ]
    }
    
    print("[~] Sending paper content to OpenRouter (openrouter/free alias)...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"[!] OpenRouter API Error [HTTP {response.status_code}]: {response.text}")
            return "openrouter/free (Failed)", "{}"
            
        result = response.json()
        model_used = result.get("model", "openrouter/free")
        content = result['choices'][0]['message']['content']
        return model_used, content
    except Exception as e:
        print(f"[!] Error during OpenRouter analysis: {e}")
        return "openrouter/free (Error)", "{}"