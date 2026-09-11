import os
import warnings
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, SYSTEM_PROMPT
from src.utils.retry import call_gemini_with_retry, QuotaExceededError
from src.llm.openrouter_client import analyze_with_openrouter

warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

def analyze_with_gemini(text_content: str) -> tuple[str, str]:
    if not GEMINI_API_KEY:
        print("[!] Error: GEMINI_API_KEY not found in environment variables.")
        return "Unknown Gemini Model", "{}"

    client = genai.Client(api_key=GEMINI_API_KEY)
    model_id = 'gemini-3.6-flash' # Aggiornato al modello corrente stabile

    print(f"[~] Sending paper content to Gemini ({model_id})...")
    try:
        response = call_gemini_with_retry(lambda: client.models.generate_content(
            model=model_id,
            contents=f"{SYSTEM_PROMPT}\n\n--- PAPER CONTENT ---\n{text_content}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        ))
        return model_id, response.text
    except QuotaExceededError as e:
        print(f"[!] Gemini free-tier quota exhausted ({e}). Falling back to OpenRouter free model...")
        return analyze_with_openrouter(text_content)
    except Exception as e:
        print(f"[!] Error during Gemini analysis: {e}")
        return model_id, "{}"
