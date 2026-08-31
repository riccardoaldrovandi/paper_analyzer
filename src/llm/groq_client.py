import os
import time
from groq import Groq
from src.config import GROQ_API_KEY, SYSTEM_PROMPT

def analyze_with_groq(text_content: str) -> tuple[str, str]:
    if not GROQ_API_KEY:
        print("[!] Error: GROQ_API_KEY not found in environment variables.")
        return "Unknown Groq Model", "{}"
    
    client = Groq(api_key=GROQ_API_KEY)
    model_id = 'llama-3.3-70b-versatile'
    
    max_chars = 20000
    if len(text_content) > max_chars:
        print(f"[~] Truncating text from {len(text_content)} to {max_chars} chars to fit Groq limits...")
        text_content = text_content[:max_chars]
    
    print("[~] Waiting 15s to ensure Groq TPM budget is clear...")
    time.sleep(15)
    
    print(f"[~] Sending paper content to Groq ({model_id})...")
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"--- PAPER CONTENT ---\n{text_content}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500
        )
        return model_id, completion.choices[0].message.content
    except Exception as e:
        print(f"[!] Error during Groq analysis: {e}")
        return model_id, "{}"
