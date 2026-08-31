import os
import warnings
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, SYSTEM_PROMPT

warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

def analyze_with_gemini(text_content: str) -> tuple[str, str]:
    if not GEMINI_API_KEY:
        print("[!] Error: GEMINI_API_KEY not found in environment variables.")
        return "Unknown Gemini Model", "{}"
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    model_id = 'gemini-2.5-flash' # Aggiornato al modello corrente stabile
    
    print(f"[~] Sending paper content to Gemini ({model_id})...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=f"{SYSTEM_PROMPT}\n\n--- PAPER CONTENT ---\n{text_content}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        return model_id, response.text
    except Exception as e:
        print(f"[!] Error during Gemini analysis: {e}")
        return model_id, "{}"
