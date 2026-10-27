"""
Paper Analyzer — versione webapp (Streamlit) di demo_pipeline_standalone.ipynb.

Stessa pipeline del notebook: INSPIRE HEP -> download PDF + sorgente LaTeX da arXiv ->
estrazione testo e figure -> digitalizzazione dei grafici (LLM con visione) ->
symbolic regression (PySR) -> analisi strutturata finale (LLM) -> confronto PDF vs LaTeX.
Provider LLM a scelta: OpenRouter (free tier), Gemini, Claude, OpenAI (con la propria API key).

Avvio:  streamlit run interactive_demo.py
"""
import os
import sys
import json
import base64
import re
import tarfile
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
import numpy as np
import requests
import pymupdf
import streamlit as st

st.set_page_config(page_title="Paper Analyzer GUI", layout="wide", page_icon="📄")

# ==========================================
# COSTANTI
# ==========================================
FREE_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
INSPIRE_URL = "https://inspirehep.net/api/literature"
PDF_DIR = "downloaded_pdfs_standalone"
LATEX_DIR = "downloaded_latex_standalone"
IMAGE_DIR = "extracted_images_standalone"
CONFIG_FILE = "config.json"
KEY_PLACEHOLDER = "YOUR_OPENROUTER_FREE_TIER_KEY_HERE"

# Provider LLM. Tutti devono supportare input immagine (la pipeline digitalizza i grafici).
# Il modello e' sempre modificabile dall'interfaccia: i default sono i modelli "standard"
# degli abbonamenti consumer. OpenRouter, OpenAI e Gemini (endpoint compatibile OpenAI)
# parlano il formato chat completions; Claude usa la Messages API nativa.
PROVIDERS = {
    "openrouter": {"label": "OpenRouter (free tier)", "model": FREE_MODEL, "config_key": "OPENROUTER_API_KEY",
                   "url": OPENROUTER_URL, "format": "openai", "paid": False, "fallback_model": None,
                   "key_help": "https://openrouter.ai/keys"},
    # Default verificato sul free tier (2026-10-26): i flash piu' recenti (3.6-3.8) rispondono 503
    # "high demand" in modo sistematico alle chiavi gratuite. Il free tier concede 5 richieste/minuto
    # per modello: `min_interval` distanzia le chiamate per restarne sotto.
    "gemini": {"label": "Google Gemini", "model": "gemini-3.5-flash", "config_key": "GEMINI_API_KEY",
               "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "format": "openai", "paid": False, "fallback_model": "gemini-3.1-flash-lite",
               "min_interval": 12.5, "key_help": "https://aistudio.google.com/apikey"},
    "claude": {"label": "Anthropic Claude", "model": "claude-sonnet-5-5", "config_key": "ANTHROPIC_API_KEY",
               "url": "https://api.anthropic.com/v1/messages", "format": "anthropic", "paid": True,
               "fallback_model": "claude-haiku-4-5-20251001", "key_help": "https://console.anthropic.com/settings/keys"},
    "openai": {"label": "OpenAI", "model": "gpt-5", "config_key": "OPENAI_API_KEY",
               "url": "https://api.openai.com/v1/chat/completions", "format": "openai", "paid": True,
               "fallback_model": None,
               "key_help": "https://platform.openai.com/api-keys"},
}
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 8192
# Errori transitori (server sovraccarico/non raggiungibile): si ritenta con attesa crescente.
TRANSIENT_STATUS = {500, 502, 503, 504, 529}
RETRY_DELAYS = (5, 15, 30, 60)   # secondi di attesa prima di ogni nuovo tentativo
# Rate limit per minuto (HTTP 429 con attesa indicata dal provider): si aspetta e si ritenta.
# Le quote giornaliere invece non si ritentano: tornano al chiamante come "quota esaurita".
RATE_LIMIT_RETRIES = 6
RATE_LIMIT_MAX_WAIT = 120        # secondi; oltre, il limite non e' "per minuto" e non vale la pena aspettare
QUESTION_TIMEOUT = 15 * 60       # se l'utente non risponde al popup l'analisi viene interrotta
MODELS_TIMEOUT = 15              # secondi per l'elenco dei modelli del provider
# Modelli che l'API elenca ma che non sono adatti all'analisi di testo + immagini.
MODEL_EXCLUDE = ("tts", "image", "live", "embedding", "transcribe", "robotics", "computer-use", "native-audio",
                 "lyria", "deep-research", "antigravity", "omni", "aqa", "whisper", "moderation", "dall-e",
                 "realtime", "audio", "instruct", "davinci", "babbage", "codex", "sora", "search")

BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
OPENROUTER_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/riccardoaldrovandi/paper_analyzer",
    "X-Title": "Thesis Paper Analyzer (standalone demo)",
}

VISION_TIMEOUT = 90
TEXT_TIMEOUT = 180
FINAL_ANALYSIS_ATTEMPTS = 3

FIELDS = [
    "problem_addressed", "main_contributions", "methodology", "datasets_used",
    "experiments", "results", "limitations", "future_work",
    "relevant_elements", "visual_analysis_findings",
]
FIELD_LABELS = {
    "problem_addressed": "🎯 Problema affrontato",
    "main_contributions": "💡 Contributi principali",
    "methodology": "🛠️ Metodologia",
    "datasets_used": "🗂️ Dataset",
    "experiments": "🧪 Esperimenti",
    "results": "📈 Risultati (dichiarati dagli autori)",
    "limitations": "⚠️ Limitazioni",
    "future_work": "🔭 Sviluppi futuri",
    "relevant_elements": "📌 Elementi rilevanti",
    "visual_analysis_findings": "🤖 Formule AI dai grafici (PySR, non degli autori)",
}

for _d in (PDF_DIR, LATEX_DIR, IMAGE_DIR):
    os.makedirs(_d, exist_ok=True)

SYSTEM_PROMPT = """
You are an expert AI research assistant. Your task is to analyze the provided text extracted from a scientific paper (which may be raw PDF text or LaTeX source code) and extract specific structured information.

You MUST return the output strictly as a valid JSON object with the following keys, and nothing else. All values must be strings or arrays of strings in English.

CRITICAL INSTRUCTION: At the very end of the paper content, you may find an "--- ADVANCED PLOT ANALYSIS ---" section. This section is NOT part of the original paper. It contains mathematical formulas independently reverse-engineered by an external curve-fitting / symbolic regression algorithm applied to the pixel data of the paper's own charts, as a way to approximate curves the authors only ever presented visually. These formulas are NOT claims, derivations, or equations stated by the paper's authors. You MUST NOT attribute them to the paper and MUST NOT merge them into "results", "methodology", or "relevant_elements" as if the authors had derived them analytically. Report them ONLY inside the dedicated "visual_analysis_findings" field below, explicitly framed as external AI-generated approximations.

JSON Schema:
{
  "problem_addressed": "Brief description of the main problem the paper tries to solve",
  "main_contributions": ["Contribution 1", "Contribution 2"],
  "methodology": "Summary of the proposed method or architecture",
  "datasets_used": ["Dataset 1", "Dataset 2"],
  "experiments": "Summary of how the models were evaluated",
  "results": "Key quantitative and qualitative results, using ONLY what the paper itself explicitly states -- never include formulas from the ADVANCED PLOT ANALYSIS section here",
  "limitations": "Any limits acknowledged by the authors",
  "future_work": "Future directions mentioned in the paper",
  "relevant_elements": "List of crucial formulas, algorithms, or tables mentioned (e.g., 'Algorithm 1: PPO Update', 'Table 3: Baseline Comparison'), taken only from the paper itself",
  "visual_analysis_findings": ["If an ADVANCED PLOT ANALYSIS section is present, one entry per curve/formula found there, each clearly worded as an AI-generated curve-fit approximation of the paper's own figure (e.g. 'AI curve-fit approximation of Figure 1, curve GQA: y = ... (not derived or stated by the paper's authors)'). Otherwise an empty list."]
}

If any information is completely missing from the text, use "Not specified" or an empty list []. Do not include markdown code block syntax (like ```json) in your response if possible, just return the raw JSON object string.
"""


# ==========================================
# LOG DI AVANZAMENTO (equivalente dei print del notebook)
# ==========================================
class RunLog:
    """Raccoglie i messaggi della pipeline e li mostra in tempo reale in un riquadro."""

    def __init__(self, placeholder=None, max_lines: int = 300):
        self.lines = []
        self.placeholder = placeholder
        self.max_lines = max_lines

    def __call__(self, msg: str):
        self.lines.append(msg)
        if self.placeholder is not None:
            self.placeholder.code("\n".join(self.lines[-self.max_lines:]), language=None)


def _noop_log(msg: str):
    pass


# ==========================================
# CONFIGURAZIONE LOCALE (API KEY)
# ==========================================
def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(config_data: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f)
    except OSError:
        pass


# ==========================================
# SUPPORTO CHIAMATE LLM (multi-provider)
# ==========================================
class QuotaExceededError(Exception):
    """Il provider ha segnalato quota esaurita o rate limit (HTTP 429)."""
    pass


class ProviderUnavailableError(Exception):
    """Il provider e' rimasto non disponibile (5xx, timeout) dopo tutti i tentativi."""
    pass


class AnalysisAborted(BaseException):
    """L'utente ha interrotto l'analisi. BaseException di proposito: i molti `except Exception`
    della pipeline (che degradano e proseguono) non devono inghiottirla."""
    pass


@dataclass
class LLM:
    """Provider + chiave + modello con cui gira l'intera pipeline.
    `log` e `on_unavailable` sono agganciati dal job in esecuzione: `on_unavailable(llm, motivo)`
    blocca finche' l'utente sceglie e ritorna "retry", "switch" (ha gia' cambiato llm.model) o "abort"."""
    provider: str
    api_key: str
    model: str
    log: object = field(default=lambda msg: None, repr=False, compare=False)
    on_unavailable: object = field(default=None, repr=False, compare=False)

    @property
    def label(self) -> str:
        return PROVIDERS[self.provider]["label"]


def _post_with_hard_timeout(url: str, headers: dict, data: str, timeout: float):
    """
    Il `timeout` di requests non copre la risoluzione DNS: la chiamata gira in un thread
    separato con un timeout "duro", così anche un resolver bloccato non blocca l'app.
    """
    box = {}

    def _worker():
        try:
            box["response"] = requests.post(url, headers=headers, data=data, timeout=timeout)
        except Exception as e:
            box["error"] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 10)
    if thread.is_alive():
        raise TimeoutError(f"Request to {url} did not complete within {timeout + 10:.0f}s (possibly stuck resolving DNS)")
    if "error" in box:
        raise box["error"]
    return box["response"]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text, expect=(dict, list)):
    """
    Estrae il primo oggetto/array JSON valido da una risposta del modello.

    Il modello free cambia a ogni chiamata e spesso non rispetta "solo JSON": aggiunge
    blocchi <think>, recinzioni ```json, frasi prima o commenti dopo. Qui si toglie tutto
    questo e si decodifica a partire dalla prima '{' o '[' che produce JSON valido del
    tipo atteso (`expect`), così un "[1]" nel preambolo non viene scambiato per il risultato.
    Solleva ValueError se non c'è JSON decodificabile.
    """
    if not text:
        raise ValueError("empty response")
    cleaned = _THINK_RE.sub("", text)
    # <think> aperto e mai chiuso (output troncato durante il ragionamento)
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        cleaned = cleaned[:cleaned.lower().index("<think>")]
    cleaned = cleaned.replace("```json", "```").replace("```JSON", "```")
    decoder = json.JSONDecoder()
    for start, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, expect) and obj:
            return obj
    raise ValueError("no valid JSON object found in the response")


def _headers(llm: LLM) -> dict:
    if PROVIDERS[llm.provider]["format"] == "anthropic":
        return {"x-api-key": llm.api_key, "anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json"}
    headers = {"Authorization": f"Bearer {llm.api_key}", "Content-Type": "application/json"}
    if llm.provider == "openrouter":
        headers.update(OPENROUTER_EXTRA_HEADERS)
    return headers


def _payload(llm: LLM, system: str, prompt: str, image_path: str = None) -> dict:
    """Corpo della richiesta: system opzionale, un messaggio utente con testo e (opzionale) un'immagine."""
    anthropic = PROVIDERS[llm.provider]["format"] == "anthropic"
    content = [{"type": "text", "text": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        if anthropic:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}})
        else:
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}})
    if anthropic:
        payload = {"model": llm.model, "max_tokens": ANTHROPIC_MAX_TOKENS, "messages": [{"role": "user", "content": content}]}
        if system:
            payload["system"] = system
        return payload
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content if image_path else prompt})
    return {"model": llm.model, "messages": messages}


_last_call = {}                  # provider -> istante (time.monotonic) dell'ultima chiamata
_last_call_lock = threading.Lock()


def _throttle(llm: LLM):
    """Rispetta `min_interval` del provider tra una chiamata e la successiva (rate limit del free tier)."""
    interval = PROVIDERS[llm.provider].get("min_interval", 0)
    if not interval:
        return
    with _last_call_lock:
        wait = _last_call.get(llm.provider, -interval) + interval - time.monotonic()
        if wait > 0:
            if wait > 2:
                llm.log(f"[~] Attendo {wait:.0f}s per restare nel limite di richieste/minuto di {llm.label}...")
            time.sleep(wait)
        _last_call[llm.provider] = time.monotonic()


def _rate_limit_wait(response) -> float:
    """
    Secondi da attendere prima di ritentare un 429, o None se non conviene ritentare.
    Gemini indica l'attesa in `RetryInfo.retryDelay` (es. "37s") e il tipo di quota in
    `QuotaFailure.violations[].quotaId`: le quote giornaliere ("PerDay") non si ritentano.
    Per gli altri provider vale l'header standard Retry-After.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    error = (body[0] if isinstance(body, list) and body else body or {})
    error = error.get("error", {}) if isinstance(error, dict) else {}
    delay = None
    for detail in (error.get("details") or []) if isinstance(error, dict) else []:
        if not isinstance(detail, dict):
            continue
        if any("perday" in str(v.get("quotaId", "")).lower() for v in detail.get("violations") or []):
            return None
        match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
        if match:
            delay = float(match.group(1))
    if delay is None:
        try:
            delay = float(response.headers.get("Retry-After", ""))
        except ValueError:
            return None
    return delay + 1 if 0 <= delay <= RATE_LIMIT_MAX_WAIT else None


def _send(llm: LLM, system: str, prompt: str, timeout: float, image_path: str = None):
    _throttle(llm)
    return _post_with_hard_timeout(PROVIDERS[llm.provider]["url"], _headers(llm),
                                   json.dumps(_payload(llm, system, prompt, image_path)), timeout=timeout)


def _send_with_retry(llm: LLM, system: str, prompt: str, timeout: float, image_path: str = None):
    """Come _send, ma ritenta 5xx/timeout/errori di connessione con attesa crescente (RETRY_DELAYS).
    Esauriti i tentativi chiede all'utente (llm.on_unavailable) cosa fare; senza callback solleva
    ProviderUnavailableError. I 429 per-minuto vengono attesi e ritentati (fino a RATE_LIMIT_RETRIES);
    se il 429 persiste (es. quota giornaliera esaurita) si chiede all'utente un altro modello, perche'
    le quote sono per modello. Senza callback i 429 e gli altri codici tornano al chiamante."""
    while True:
        reason, invalid, rate_limited = "", False, 0
        for attempt, delay in enumerate((0,) + RETRY_DELAYS):
            if delay:
                llm.log(f"[~] {llm.label} non disponibile ({reason[:120]}). Riprovo tra {delay}s "
                        f"(tentativo {attempt}/{len(RETRY_DELAYS)})...")
                time.sleep(delay)
            try:
                response = _send(llm, system, prompt, timeout, image_path)
                while response.status_code == 429 and rate_limited < RATE_LIMIT_RETRIES:
                    wait = _rate_limit_wait(response)
                    if wait is None:
                        break
                    rate_limited += 1
                    llm.log(f"[~] {llm.label}: limite di richieste al minuto raggiunto. Attendo {wait:.0f}s "
                            f"e riprovo ({rate_limited}/{RATE_LIMIT_RETRIES})...")
                    time.sleep(wait)
                    response = _send(llm, system, prompt, timeout, image_path)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, TimeoutError) as e:
                reason = f"{type(e).__name__}: {e}"
                continue
            if response.status_code == 404 and llm.on_unavailable is not None:
                # modello inesistente o non piu' disponibile: inutile ritentare, si chiede un altro modello
                reason = f"HTTP 404: {' '.join(response.text.split())[:300]}"
                invalid = True
                break
            if response.status_code == 429 and llm.on_unavailable is not None:
                # quota del modello esaurita: ritentare lo stesso modello non serve, un altro ha la sua quota
                reason = f"HTTP 429: {' '.join(response.text.split())[:300]}"
                invalid = True
                break
            if response.status_code in TRANSIENT_STATUS:
                reason = f"HTTP {response.status_code}: {' '.join(response.text.split())[:200]}"
                continue
            return response
        if invalid and reason.startswith("HTTP 429"):
            llm.log(f"[!] Quota di {llm.model} su {llm.label} esaurita: {reason[:300]}")
        elif invalid:
            llm.log(f"[!] Il modello {llm.model} non è utilizzabile su {llm.label}: {reason[:300]}")
        else:
            llm.log(f"[!] {llm.label} ancora non disponibile dopo {len(RETRY_DELAYS)} tentativi: {reason[:200]}")
        if llm.on_unavailable is None:
            raise ProviderUnavailableError(reason)
        decision = llm.on_unavailable(llm, reason, invalid)
        if decision not in ("retry", "switch") or (invalid and decision != "switch"):
            raise AnalysisAborted()


def list_models(llm: LLM) -> list:
    """ID dei modelli di chat/visione che il provider offre alla chiave dell'utente. [] se non disponibile."""
    try:
        if llm.provider == "gemini":
            names, token = [], None
            for _ in range(5):  # al massimo 5 pagine
                params = {"pageSize": 200, **({"pageToken": token} if token else {})}
                resp = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params=params,
                                    headers={"x-goog-api-key": llm.api_key}, timeout=MODELS_TIMEOUT)
                if resp.status_code != 200:
                    return []
                body = resp.json()
                names += [m["name"].split("/", 1)[-1] for m in body.get("models", [])
                          if "generateContent" in m.get("supportedGenerationMethods", [])]
                token = body.get("nextPageToken")
                if not token:
                    break
        elif llm.provider == "openai":
            resp = requests.get("https://api.openai.com/v1/models", headers=_headers(llm), timeout=MODELS_TIMEOUT)
            if resp.status_code != 200:
                return []
            names = [m["id"] for m in resp.json().get("data", []) if re.match(r"(gpt|chatgpt|o\d)", m["id"])]
        elif llm.provider == "claude":
            resp = requests.get("https://api.anthropic.com/v1/models", params={"limit": 100}, headers=_headers(llm),
                                timeout=MODELS_TIMEOUT)
            if resp.status_code != 200:
                return []
            names = [m["id"] for m in resp.json().get("data", [])]
        else:
            return []
    except Exception:
        return []
    return sorted({n for n in names if not any(x in n.lower() for x in MODEL_EXCLUDE)})


def _model_version(name: str) -> tuple:
    return tuple(float(x) for x in re.findall(r"\d+(?:\.\d+)?", name) if float(x) < 100)


def _model_family(name: str) -> str:
    return next((f for f in ("lite", "flash", "pro", "haiku", "sonnet", "opus", "mini") if f in name.lower()), "")


def rank_alternatives(current: str, models: list) -> tuple:
    """(elenco ordinato, consigliato). Consigliato = stessa famiglia del modello in uso, versione stabile
    immediatamente inferiore. Per un alias ("-latest") si assume che punti alla versione piu' alta della famiglia."""
    family = _model_family(current)
    same = [m for m in models if m != current and _model_family(m) == family and "latest" not in m]
    stable = [m for m in same if "preview" not in m and "exp" not in m] or same
    cur_v = _model_version(current)
    if "latest" in current or not cur_v:
        cur_v = max((_model_version(m) for m in stable), default=())
    below = sorted((m for m in stable if _model_version(m) < cur_v), key=_model_version, reverse=True)
    suggested = (below or sorted(stable, key=_model_version, reverse=True) or [None])[0]
    rest = sorted((m for m in models if m != current and m != suggested),
                  key=lambda m: (_model_family(m) != family, "preview" in m, tuple(-v for v in _model_version(m))))
    return ([suggested] if suggested else []) + rest, suggested


def _answer(llm: LLM, result: dict) -> tuple:
    """(testo della risposta, modello effettivo). Solleva RuntimeError se la risposta e' vuota."""
    if PROVIDERS[llm.provider]["format"] == "anthropic":
        text = "".join(b.get("text", "") for b in result.get("content") or [] if b.get("type") == "text")
        finish = result.get("stop_reason")
    else:
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"{llm.label} returned no choices: {json.dumps(result)[:500]}")
        text = (choices[0].get("message") or {}).get("content") or ""
        finish = choices[0].get("finish_reason")
    if not text.strip():
        raise RuntimeError(f"{llm.label} returned an empty answer (finish_reason={finish})")
    return text, result.get("model", llm.model)


# ==========================================
# ANALISI TESTO
# ==========================================
def analyze_with_llm(llm: LLM, text_content: str, log=_noop_log) -> dict:
    """
    Analisi strutturata finale. Ritorna un dict con:
      model (str), data (dict o None), raw (risposta grezza), error (str o None), attempts (int).
    Se la risposta non e' JSON valido riprova (l'alias free di OpenRouter instrada a modelli diversi).
    """
    outcome = {"model": llm.model, "data": None, "raw": "", "error": None, "attempts": 0}

    for attempt in range(1, FINAL_ANALYSIS_ATTEMPTS + 1):
        outcome["attempts"] = attempt
        log(f"[~] Sending paper content to {llm.label} ({llm.model}), attempt {attempt}/{FINAL_ANALYSIS_ATTEMPTS}...")
        try:
            response = _send_with_retry(llm, SYSTEM_PROMPT, f"--- PAPER CONTENT ---\n{text_content}", TEXT_TIMEOUT)
            if response.status_code == 429:
                outcome["error"] = f"{llm.label}: quota exhausted or rate limit (HTTP 429): {response.text[:500]}"
                log(f"[!] {outcome['error']}")
                return outcome
            if response.status_code in (401, 403):
                outcome["error"] = f"{llm.label} rejected the API key (HTTP {response.status_code}): {response.text[:500]}"
                log(f"[!] {outcome['error']}")
                return outcome
            if response.status_code != 200:
                outcome["error"] = f"{llm.label} API Error [HTTP {response.status_code}]: {response.text[:500]}"
                log(f"[!] {outcome['error']}")
                continue

            result = response.json()
            if "error" in result and not (result.get("choices") or result.get("content")):
                outcome["error"] = f"{llm.label} error: {json.dumps(result['error'])[:500]}"
                log(f"[!] {outcome['error']}")
                continue

            outcome["raw"], outcome["model"] = _answer(llm, result)
            outcome["data"] = extract_json(outcome["raw"], expect=dict)
            outcome["error"] = None
            log(f"[+] Structured analysis received (model: {outcome['model']}).")
            return outcome
        except ValueError as e:
            outcome["error"] = f"The model ({outcome['model']}) did not return valid JSON: {e}"
            log(f"[!] {outcome['error']}")
        except ProviderUnavailableError as e:
            outcome["error"] = f"{llm.label} non disponibile dopo i tentativi: {e}"
            log(f"[!] {outcome['error']}")
            return outcome
        except Exception as e:
            outcome["error"] = f"Error during {llm.label} analysis: {e}"
            log(f"[!] {outcome['error']}")

    return outcome


# ==========================================
# ANALISI IMMAGINI (Vision)
# ==========================================
def analyze_image_with_llm(llm: LLM, image_path: str, prompt: str) -> str:
    response = _send_with_retry(llm, None, prompt, VISION_TIMEOUT, image_path=image_path)
    if response.status_code == 429:
        raise QuotaExceededError(response.text[:500])
    if response.status_code != 200:
        raise RuntimeError(f"{llm.label} API Error [HTTP {response.status_code}]: {response.text[:500]}")
    return _answer(llm, response.json())[0]


# ==========================================
# DOWNLOAD E RICERCA
# ==========================================
def sanitize_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|\n\r\t]', "", title)[:50].strip()


def download_pdf(pdf_url: str, paper_id: str, title: str, log=_noop_log) -> str:
    if not pdf_url or pdf_url == "N/A":
        return "N/A"
    filepath = os.path.join(PDF_DIR, f"{str(paper_id)[:8]}_{sanitize_filename(title)}.pdf")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        log(f"[-] PDF already exists locally, skipping download: {os.path.basename(filepath)}")
        return filepath
    try:
        log(f"[~] Attempting to download PDF: {pdf_url}")
        response = requests.get(pdf_url, headers=BROWSER_HEADERS, timeout=30)
        is_pdf = "application/pdf" in response.headers.get("Content-Type", "") or response.content[:5] == b"%PDF-"
        if response.status_code == 200 and is_pdf:
            tmp_path = filepath + ".part"
            with open(tmp_path, "wb") as f:
                f.write(response.content)
            os.replace(tmp_path, filepath)
            log(f"[+] Saved PDF to: {os.path.basename(filepath)}")
            return filepath
        log(f"[!] The link didn't point directly to a PDF (Status: {response.status_code})")
    except Exception as e:
        log(f"[!] PDF download failed: {e}")
    return "N/A"


def download_latex(arxiv_id: str, paper_id: str, title: str, log=_noop_log) -> str:
    if not arxiv_id or arxiv_id == "N/A":
        return "N/A"
    filepath = os.path.join(LATEX_DIR, f"{str(paper_id)[:8]}_{sanitize_filename(title)}_source.tar.gz")
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        log(f"[-] LaTeX source already exists locally, skipping download: {os.path.basename(filepath)}")
        return filepath
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        log(f"[~] Attempting to download LaTeX source: {url}")
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
        if response.status_code != 200:
            log(f"[!] Failed to get LaTeX source (Status: {response.status_code})")
            return "N/A"
        # /e-print/ serve un PDF (non un tar.gz) quando l'autore non ha caricato il sorgente
        content_type = response.headers.get("Content-Type", "")
        if not ("gzip" in content_type or response.content[:2] == b"\x1f\x8b"):
            log(f"[!] No real LaTeX source for {arxiv_id}: arXiv served '{content_type or 'non-gzip'}' (likely PDF-only submission).")
            return "N/A"
        tmp_path = filepath + ".part"
        with open(tmp_path, "wb") as f:
            f.write(response.content)
        os.replace(tmp_path, filepath)
        log(f"[+] Saved LaTeX source to: {os.path.basename(filepath)}")
        return filepath
    except Exception as e:
        log(f"[!] LaTeX download failed: {e}")
    return "N/A"


def search_inspire_hep(query: str, limit: int, min_citations: int, sort_by, log=_noop_log) -> list:
    """Interroga INSPIRE HEP (API pubblica, nessuna chiave). Ritorna solo i metadati: nessun download."""
    params = {"q": query, "size": limit, "fields": "titles,authors.full_name,author_count,arxiv_eprints,citation_count,earliest_date"}
    if sort_by:
        params["sort"] = sort_by
    log(f"[INSPIRE HEP] Searching for: '{query}'")
    try:
        response = requests.get(INSPIRE_URL, params=params, timeout=30)
    except Exception as e:
        raise RuntimeError(f"INSPIRE HEP unreachable: {e}")
    if response.status_code != 200:
        raise RuntimeError(f"INSPIRE API Error: HTTP {response.status_code}")

    papers = []
    for item in response.json().get("hits", {}).get("hits", []):
        metadata = item.get("metadata", {})
        titles = metadata.get("titles") or [{}]
        title = titles[0].get("title", "Unknown Title")
        citations = metadata.get("citation_count", 0) or 0
        if citations < min_citations:
            log(f"[-] Discarded: '{title}' (Citations: {citations} < {min_citations})")
            continue
        eprints = metadata.get("arxiv_eprints") or [{}]
        authors = [a.get("full_name", "?") for a in (metadata.get("authors") or [])[:3]]
        if (metadata.get("author_count") or 0) > len(authors):
            authors.append("et al.")
        papers.append({
            "Paper_ID": str(item.get("id", "unknown")), "Title": title, "Citations": citations,
            "ArXiv_ID": eprints[0].get("value", "N/A"), "Authors": ", ".join(authors),
            "Year": (metadata.get("earliest_date") or "")[:4],
        })
    return papers


def download_sources(paper: dict, want_pdf: bool, want_latex: bool, log=_noop_log) -> dict:
    """Scarica PDF e/o sorgente LaTeX del paper scelto. Ritorna una copia con i percorsi locali ('N/A' se mancanti)."""
    arxiv_id = paper["ArXiv_ID"]
    out = {**paper, "Local_PDF_Path": "N/A", "Local_LaTeX_Path": "N/A"}
    if arxiv_id == "N/A":
        return out
    if want_pdf:
        out["Local_PDF_Path"] = download_pdf(f"https://arxiv.org/pdf/{arxiv_id}.pdf", paper["Paper_ID"], paper["Title"], log)
    if want_latex:
        if want_pdf:
            time.sleep(2)  # cortesia verso arXiv
        out["Local_LaTeX_Path"] = download_latex(arxiv_id, paper["Paper_ID"], paper["Title"], log)
    return out


# ==========================================
# ESTRAZIONE TESTO
# ==========================================
def extract_text_from_pdf(pdf_path: str) -> str:
    with pymupdf.open(pdf_path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def extract_text_from_latex_tarball(tar_path: str, log=_noop_log) -> str:
    tex_content = []
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        raw_bytes = f.read()
                        try:
                            content = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            content = raw_bytes.decode("latin-1")
                        tex_content.append(f"--- SOURCE FILE: {member.name} ---\n{content}\n")
    except Exception as e:
        log(f"[!] Error reading LaTeX archive {tar_path}: {e}")
    return "\n".join(tex_content)


# ==========================================
# ESTRAZIONE FIGURE
# ==========================================
def _figure_dir_for(source_path: str) -> str:
    """Stessa cartella per PDF e LaTeX dello stesso paper (come nel notebook)."""
    base_name = os.path.basename(source_path)
    if base_name.endswith(".tar.gz"):
        base_name = base_name[:-7]
    elif base_name.endswith(".pdf"):
        base_name = base_name[:-4]
    if base_name.endswith("_source"):
        base_name = base_name[:-7]
    return os.path.join(IMAGE_DIR, base_name)


def _is_actual_plot(llm: LLM, image_path: str, log=_noop_log) -> bool:
    prompt = (
        "Analyze this document page image. Does this page contain a primary data plot, "
        "statistical chart, bar chart, line graph, or mathematical curve with coordinate axes? "
        "Answer ONLY with 'YES' if it features a data plot/chart, or 'NO' if it is just text, "
        "tables, or architectural block diagrams without plotted data curves."
    )
    try:
        answer = _THINK_RE.sub("", analyze_image_with_llm(llm, image_path, prompt)).strip().upper()
        return "YES" in answer
    except QuotaExceededError:
        raise
    except Exception as e:
        log(f"[!] Error during smart filtering (keeping the page): {e}")
        return True  # nel dubbio la pagina si tiene


def _crop_individual_plots(llm: LLM, page, page_filepath: str, output_dir: str, base_name: str, page_index: int, log=_noop_log) -> list:
    prompt = (
        "Analyze this document page. Locate every distinct data plot, statistical chart, "
        "bar chart, or line graph present on the page. "
        "For each plot, provide its bounding box coordinates [ymin, xmin, ymax, xmax] scaled from 0 to 1000. "
        "You MUST return ONLY a valid JSON array of objects. Example: "
        '[{"box_2d": [100, 100, 300, 400], "label": "fig5"}]'
    )
    try:
        boxes = extract_json(analyze_image_with_llm(llm, page_filepath, prompt))
        if isinstance(boxes, dict):
            boxes = [boxes]
        cropped_images = []
        rect = page.rect
        for i, item in enumerate(boxes if isinstance(boxes, list) else []):
            if not isinstance(item, dict):
                continue
            box = item.get("box_2d")
            try:
                ymin, xmin, ymax, xmax = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                continue
            if (ymax - ymin) < 30 or (xmax - xmin) < 30:  # box troppo piccolo per essere un grafico
                continue

            y0, x0 = ymin / 1000.0 * rect.height, xmin / 1000.0 * rect.width
            y1, x1 = ymax / 1000.0 * rect.height, xmax / 1000.0 * rect.width
            clip_rect = pymupdf.Rect(max(0, x0 - 10), max(0, y0 - 10), min(rect.width, x1 + 10), min(rect.height, y1 + 10))
            pix = page.get_pixmap(dpi=300, clip=clip_rect)
            label = sanitize_filename(str(item.get("label", f"plot{i + 1}"))).replace(" ", "_") or f"plot{i + 1}"
            crop_path = os.path.join(output_dir, f"{base_name}_page{page_index + 1}_crop{i + 1}_{label}.png")
            pix.save(crop_path)
            cropped_images.append(crop_path)

        if cropped_images:
            if os.path.exists(page_filepath):
                os.remove(page_filepath)
            return cropped_images
        log(f"[-] No usable bounding box on page {page_index + 1}, keeping the full page.")
    except QuotaExceededError:
        raise
    except Exception as e:
        log(f"[!] Error during smart cropping, falling back to full page: {e}")
    return [page_filepath]


def extract_images_from_pdf(llm: LLM, pdf_path: str, log=_noop_log) -> list:
    output_dir = _figure_dir_for(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(output_dir)
    saved_images = []

    log("[~] Scanning PDF and applying smart filtering & cropping to charts...")
    with pymupdf.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]

            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    base_image = doc.extract_image(img[0])
                except Exception:
                    continue
                if not base_image or base_image["width"] < 200 or base_image["height"] < 200:
                    continue
                if base_image["ext"] not in ("png", "jpg", "jpeg"):
                    continue  # formati (jpx, jbig2...) che i modelli vision non accettano
                image_filepath = os.path.join(output_dir, f"{base_name}_page{page_index + 1}_raster{img_index + 1}.{base_image['ext']}")
                with open(image_filepath, "wb") as f:
                    f.write(base_image["image"])
                saved_images.append(image_filepath)

            page_text = page.get_text("text").lower()
            if "fig" in page_text or "plot" in page_text:
                page_filepath = os.path.join(output_dir, f"{base_name}_page{page_index + 1}_full_render.png")
                page.get_pixmap(dpi=200).save(page_filepath)
                log(f"[~] Visual page analysis {page_index + 1} via {llm.label} Vision...")
                try:
                    if _is_actual_plot(llm, page_filepath, log):
                        log(f"[+] Found valid chart(s) at page {page_index + 1}. Cropping individual plots...")
                        saved_images.extend(_crop_individual_plots(llm, page, page_filepath, output_dir, base_name, page_index, log))
                    else:
                        log(f"[-] Page {page_index + 1} mentions a figure but is not a numerical plot. Removed.")
                        os.remove(page_filepath)
                except QuotaExceededError as e:
                    log(f"[!] {llm.label}: quota exhausted or rate limit ({e}). Stopping figure extraction for this PDF.")
                    saved_images.append(page_filepath)
                    break
    return saved_images


def extract_images_from_latex(tar_path: str, log=_noop_log) -> list:
    output_dir = _figure_dir_for(tar_path)
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".pdf", ".eps")

    log("[~] Extracting images from LaTeX archive...")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                name_lower = member.name.lower()
                if not (member.isfile() and name_lower.endswith(valid_extensions)):
                    continue
                try:
                    # filter="data": blocca path traversal / link pericolosi in archivi non fidati
                    tar.extract(member, path=output_dir, filter="data")
                    extracted_path = os.path.join(output_dir, member.name)

                    if name_lower.endswith(".pdf"):
                        png_path = extracted_path[:-4] + ".png"
                        with pymupdf.open(extracted_path) as plot_doc:
                            plot_doc[0].get_pixmap(dpi=300).save(png_path)
                        os.remove(extracted_path)
                        saved_images.append(png_path)
                    elif name_lower.endswith(".eps"):
                        # PyMuPDF non legge EPS: conversione diretta con Ghostscript
                        png_path = extracted_path[:-4] + ".png"
                        subprocess.run(
                            ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r300", "-dEPSCrop",
                             f"-o{png_path}", extracted_path],
                            check=True, capture_output=True, timeout=30,
                        )
                        os.remove(extracted_path)
                        saved_images.append(png_path)
                    else:
                        saved_images.append(extracted_path)
                except FileNotFoundError as e:
                    if "gs" in str(e):
                        log(f"[!] Cannot convert {member.name}: 'gs' (Ghostscript) not found on the PATH.")
                    else:
                        log(f"[!] Error extracting {member.name}: {e}")
                except Exception as e:
                    log(f"[!] Error extracting/converting {member.name}: {e}")
    except Exception as e:
        log(f"[!] Error during processing of {tar_path}: {e}")
    return saved_images


# ==========================================
# DIGITALIZZAZIONE GRAFICI (Vision)
# ==========================================
VISION_DIGITIZE_PROMPT = """
You are an expert data extraction algorithm. Analyze this plot image carefully.

Step 1: Determine if the plot is "continuous" (one or more line/curve trends showing
a mathematical relationship) or "discrete" (a bar chart or scatter plot comparing
independent categories/models with no continuous trend).

Step 2 (continuous plots only): This plot may contain MULTIPLE distinct curves
(e.g. differently colored or dashed lines identified by a legend). You MUST keep
them separate. For EACH distinct curve, extract 15 to 30 (X, Y) numerical float
coordinates sampled along that single curve, and give it a "label" taken from the
legend (or a short visual description like "upper line" if there is no legend).
Never mix points belonging to different curves into the same series.

Step 3 (discrete plots only): extract each individual data point (X, Y) and its
corresponding text label read directly from the plot.

You MUST return ONLY a valid JSON object. Do not include markdown blocks.
Format exactly like this:
{
    "plot_type": "continuous" or "discrete",
    "series": [
        {"label": "curve or point label", "points": [[x1, y1], [x2, y2]]}
    ],
    "summary": "For discrete plots only: a brief 1-sentence analytical summary of what the points show (e.g. 'Model X has higher accuracy but is slower...'). Leave empty for continuous plots."
}

For continuous plots, create one "series" entry per distinct curve found (usually
1 to 4). For discrete plots, create one "series" entry per data point, each with a
single-element "points" list.
"""


def _valid_points(points) -> list:
    """Tiene solo le coppie [x, y] numeriche e finite (il modello a volte restituisce stringhe o terne)."""
    clean = []
    for p in points if isinstance(points, list) else []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            clean.append([x, y])
    return clean


def _clean_points(points: list) -> tuple:
    """Ordina per X e rimuove le X duplicate (tiene la prima occorrenza)."""
    arr = np.array(points, dtype=float)
    arr = arr[arr[:, 0].argsort()]
    _, unique_idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(unique_idx)]
    return arr[:, 0], arr[:, 1]


def digitize_plot(llm: LLM, image_path: str, log=_noop_log):
    log(f"[~] Computer vision analysis in progress: {os.path.basename(image_path)}")
    raw_output = ""
    try:
        raw_output = analyze_image_with_llm(llm, image_path, VISION_DIGITIZE_PROMPT)
        data = extract_json(raw_output, expect=dict)
        raw_series = [s for s in (data.get("series") or []) if isinstance(s, dict)]
        plot_type = data.get("plot_type", "continuous")
        if plot_type not in ("continuous", "discrete"):
            plot_type = "continuous"
        data["plot_type"] = plot_type

        if plot_type == "continuous":
            parsed_series = []
            for entry in raw_series:
                points = _valid_points(entry.get("points"))
                if len(points) < 2:
                    continue
                X, y = _clean_points(points)
                if len(X) < 2:
                    continue
                parsed_series.append({"label": str(entry.get("label", f"curve{len(parsed_series) + 1}")), "X": X, "y": y})
            if not parsed_series:
                log(f"[-] {llm.label} did not find enough points in any curve.")
                return None
            data["series"] = parsed_series
            log(f"[+] Detected '{plot_type}' plot with {len(parsed_series)} distinct curve(s).")
        else:
            labels, all_points = [], []
            for i, entry in enumerate(raw_series):
                points = _valid_points(entry.get("points"))
                if points:
                    all_points.append(points[0])
                    labels.append(str(entry.get("label", f"item{i + 1}")))
            if not all_points:
                log(f"[-] {llm.label} did not find any data points in the graph.")
                return None
            data_matrix = np.array(all_points)
            data["X"], data["y"], data["labels"] = data_matrix[:, 0], data_matrix[:, 1], labels
            log(f"[+] Detected '{plot_type}' plot with {len(all_points)} data points.")
        return data
    except QuotaExceededError:
        raise
    except ValueError as e:
        log(f"[-] {llm.label} returned a non-JSON output ({e}). Raw response (truncated):\n{raw_output[:1000]}")
        return None
    except Exception as e:
        log(f"[-] Error during Vision API call: {e}")
        return None


# ==========================================
# SYMBOLIC REGRESSION (PySR) IN UN PROCESSO SEPARATO
# ==========================================
# PySR/Julia gira in un sottoprocesso: Julia dentro il thread di Streamlit non è
# affidabile, e un crash di Julia non deve far cadere la webapp. Tutte le curve di
# una sorgente vanno in un solo sottoprocesso, così Julia si avvia una volta sola.
# Stessa logica del notebook: operatori + - * / exp log, holdout 25% se n >= 8,
# rasoio di Occam tra le equazioni entro 0.02 dal miglior R^2 di holdout.
PYSR_WORKER = r'''
import json, sys
import numpy as np

in_path, out_path = sys.argv[1], sys.argv[2]
with open(in_path) as f:
    job = json.load(f)

results = []

def dump():
    with open(out_path, "w") as f:
        json.dump(results, f)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    scale = max(abs(np.mean(y_true)), 1.0)
    if ss_tot > 1e-8 * (scale ** 2) * len(y_true):
        return float(1 - ss_res / ss_tot)
    rel_err = np.sqrt(ss_res / len(y_true)) / scale
    return float(max(0.0, 1.0 - rel_err))

try:
    from pysr import PySRRegressor
except Exception as e:
    results = [{"equation": "PySR not available", "error": str(e)} for _ in job["curves"]]
    dump()
    sys.exit(0)

for curve in job["curves"]:
    try:
        X = np.asarray(curve["X"], dtype=float).reshape(-1, 1)
        y = np.asarray(curve["y"], dtype=float).reshape(-1)
        model = PySRRegressor(
            niterations=job["iterations"],
            binary_operators=["+", "*", "/", "-"],
            unary_operators=["exp", "log"],
            model_selection="best",
            parsimony=0.01,
            nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"log": 0, "exp": 0}},
            verbosity=0,
            progress=False,
            temp_equation_file=True,
        )
        n = len(y)
        use_holdout = n >= 8
        if use_holdout:
            idx = np.random.default_rng(42).permutation(n)
            n_holdout = min(max(3, round(n * 0.25)), n - 5)
            hold_idx, fit_idx = idx[:n_holdout], idx[n_holdout:]
            X_fit, y_fit, X_hold, y_hold = X[fit_idx], y[fit_idx], X[hold_idx], y[hold_idx]
        else:
            X_fit, y_fit = X, y

        model.fit(X_fit, y_fit)

        best_index, r2_holdout = None, None
        if use_holdout:
            scored = []
            for index in model.equations_.index:
                y_pred = np.asarray(model.predict(X_hold, index=index)).reshape(-1)
                scored.append((index, r2_score(y_hold, y_pred), model.equations_.loc[index, "complexity"]))
            best_r2 = max(s[1] for s in scored)
            best_index, r2_holdout, _ = min([s for s in scored if s[1] >= best_r2 - 0.02], key=lambda s: s[2])

        best_row = model.equations_.loc[best_index] if best_index is not None else model.get_best()
        r2_fit = r2_score(y_fit, np.asarray(model.predict(X_fit, index=best_index)).reshape(-1))
        results.append({
            "equation": str(model.sympy(index=best_index)),
            "complexity": int(best_row["complexity"]),
            "score": float(best_row["score"]),
            "r2": r2_fit,
            "r2_holdout": r2_holdout,
            "n_points": n,
        })
    except Exception as e:
        results.append({"equation": "Error in PySR calculation", "error": str(e)})
    dump()  # salvataggio incrementale: in caso di timeout restano le curve già fatte
dump()
'''


def run_pysr_batch(curves: list, iterations: int, log=_noop_log) -> list:
    """Esegue PySR su tutte le curve in un sottoprocesso. Ritorna un risultato per curva."""
    if not curves:
        return []
    failed = lambda msg: {"equation": msg, "error": msg}
    timeout = 600 + 240 * len(curves)  # il primo avvio compila Julia e può durare minuti

    with tempfile.TemporaryDirectory(prefix="pysr_job_") as tmp_dir:
        in_path = os.path.join(tmp_dir, "in.json")
        out_path = os.path.join(tmp_dir, "out.json")
        script_path = os.path.join(tmp_dir, "pysr_worker.py")
        with open(in_path, "w") as f:
            json.dump({"iterations": iterations, "curves": [{"X": np.asarray(c["X"]).tolist(), "y": np.asarray(c["y"]).tolist()} for c in curves]}, f)
        with open(script_path, "w") as f:
            f.write(PYSR_WORKER)

        env = dict(os.environ, PYTHON_JULIACALL_HANDLE_SIGNALS="yes")
        log(f"[~] Starting PySR (Symbolic Regression) on {len(curves)} curve(s), {iterations} iterations each "
            f"(timeout {timeout // 60} min; the first run compiles Julia and is slower)...")
        status_msg = None
        try:
            proc = subprocess.run([sys.executable, script_path, in_path, out_path], cwd=tmp_dir, env=env,
                                  capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip()[-800:]
                status_msg = f"PySR process exited with code {proc.returncode}"
                log(f"[!] {status_msg}. stderr (tail):\n{tail}")
        except subprocess.TimeoutExpired:
            status_msg = "PySR timed out"
            log(f"[!] {status_msg} after {timeout}s.")

        results = []
        try:
            with open(out_path) as f:
                results = json.load(f)
        except Exception:
            pass

    results = results[:len(curves)]
    results += [failed(status_msg or "PySR produced no result")] * (len(curves) - len(results))
    for c, r in zip(curves, results):
        if r.get("error"):
            log(f"[-] Curve '{c['label']}': {r['equation']} ({r['error'][:200]})")
        else:
            log(f"[+] Curve '{c['label']}': y = {r['equation']}")
    return results


def _confidence_note(result: dict) -> str:
    if result.get("r2_holdout") is not None:
        note = f", R^2 fit={result['r2']:.3f}, holdout={result['r2_holdout']:.3f}"
        if result["r2_holdout"] < 0.85 or (result["r2"] - result["r2_holdout"]) > 0.15:
            note += " (low confidence, possible overfitting)"
        return note
    if result.get("r2") is not None:
        return f", R^2={result['r2']:.3f} (not cross-validated, too few points)"
    return ""


# ==========================================
# PIPELINE COMBINATA
# ==========================================
def build_enriched_paper_text(llm: LLM, source_path: str, kind: str, iterations: int, max_figures: int, log=_noop_log) -> dict:
    """
    Testo del paper + sezione "ADVANCED PLOT ANALYSIS" (come nel notebook).
    max_figures = 0 significa nessun limite.
    """
    out = {"text": "", "figures": [], "analyzed_figures": [], "advanced_blocks": [], "warnings": []}

    log(f"[1/4] Extracting text from {kind.upper()}: {os.path.basename(source_path)}")
    out["text"] = extract_text_from_pdf(source_path) if kind == "pdf" else extract_text_from_latex_tarball(source_path, log)
    if not out["text"].strip():
        out["warnings"].append("Nessun testo estratto dalla sorgente.")
        log("[!] No text extracted from the source.")

    log("[2/4] Extracting figures...")
    image_paths = extract_images_from_pdf(llm, source_path, log) if kind == "pdf" else extract_images_from_latex(source_path, log)
    out["figures"] = image_paths
    log(f"[+] Found {len(image_paths)} figure(s).")
    if max_figures > 0 and len(image_paths) > max_figures:
        log(f"[!] Capping analysis to the first {max_figures} figure(s) out of {len(image_paths)}.")
        image_paths = image_paths[:max_figures]
    out["analyzed_figures"] = image_paths

    log(f"[3/4] Digitizing figures via {llm.label} Vision...")
    digitized = []
    for i, image_path in enumerate(image_paths):
        log(f"--- Figure {i + 1}/{len(image_paths)} ---")
        try:
            plot_data = digitize_plot(llm, image_path, log)
        except QuotaExceededError as e:
            msg = f"Quota/rate limit {llm.label} esauriti durante la digitalizzazione: analizzate {i}/{len(image_paths)} figure."
            out["warnings"].append(msg)
            log(f"[!] {llm.label}: quota exhausted or rate limit ({e}). Stopping figure analysis here.")
            break
        if plot_data is None:
            log(f"[!] Skipping {os.path.basename(image_path)}: could not extract data from this image.")
            continue
        digitized.append((image_path, plot_data))

    curves = [{"label": s["label"], "X": s["X"], "y": s["y"]}
              for _, d in digitized if d["plot_type"] == "continuous" for s in d["series"]]
    log("[4/4] Symbolic regression on the continuous curves...")
    formulas = iter(run_pysr_batch(curves, iterations, log))

    for image_path, plot_data in digitized:
        lines = [f"[Figure: {os.path.basename(image_path)}] ({plot_data['plot_type']} plot)"]
        if plot_data["plot_type"] == "continuous":
            for series in plot_data["series"]:
                result = next(formulas)
                if result.get("error"):
                    continue
                lines.append(f"  - Curve '{series['label']}': y = {result['equation']}{_confidence_note(result)}")
        else:
            if plot_data.get("summary"):
                lines.append(f"  Summary: {plot_data['summary']}")
            labels = plot_data.get("labels", [])
            for i, (px, py) in enumerate(zip(plot_data["X"], plot_data["y"])):
                label = labels[i] if i < len(labels) else f"Item {i + 1}"
                lines.append(f"  - {label}: ({px}, {py})")
        if len(lines) > 1:
            out["advanced_blocks"].append("\n".join(lines))

    if out["advanced_blocks"]:
        disclaimer = (
            "(The formulas below were derived by an independent AI curve-fitting/symbolic-regression "
            "tool applied to the pixel data of each figure. They are NOT part of the original paper and "
            "were NOT derived or stated by its authors -- report them separately, never as the paper's own results.)"
        )
        out["text"] += "\n\n--- ADVANCED PLOT ANALYSIS ---\n" + disclaimer + "\n\n" + "\n\n".join(out["advanced_blocks"])
        log(f"[+] Enriched paper content with analysis of {len(out['advanced_blocks'])}/{len(image_paths)} figure(s).")
    else:
        log("[!] No usable figure analysis produced; proceeding with text-only content.")
    return out


def run_source(llm: LLM, source_path: str, kind: str, iterations: int, max_figures: int, log) -> dict:
    """Pipeline completa per una sorgente. Non solleva mai: gli errori finiscono in 'error'."""
    result = {"source": source_path, "error": None, "warnings": [], "figures": [], "analyzed_figures": [],
              "advanced_blocks": [], "text_chars": 0, "model": None, "data": None, "raw": "", "attempts": 0}
    try:
        enriched = build_enriched_paper_text(llm, source_path, kind, iterations, max_figures, log)
        result.update({k: enriched[k] for k in ("figures", "analyzed_figures", "advanced_blocks", "warnings")})
        result["text_chars"] = len(enriched["text"])
        result["enriched_text"] = enriched["text"]
        log(f"Enriched text length ({kind.upper()}): {result['text_chars']} characters")
        if not enriched["text"].strip():
            result["error"] = "Nessun testo da analizzare."
            return result
        analysis = analyze_with_llm(llm, enriched["text"], log)
        result.update({k: analysis[k] for k in ("model", "data", "raw", "attempts")})
        result["error"] = analysis["error"] if analysis["data"] is None else None
    except Exception as e:
        result["error"] = f"Errore inatteso nella pipeline {kind.upper()}: {type(e).__name__}: {e}"
        log(f"[!] {result['error']}")
    return result


@st.cache_resource
def _pipeline_lock() -> threading.Lock:
    """Un'analisi alla volta per tutto il server: PyMuPDF non supporta l'uso concorrente da più thread."""
    return threading.Lock()


# ==========================================
# JOB IN BACKGROUND
# ==========================================
# La pipeline gira in un thread: gli script Streamlit ripartono a ogni interazione (anche col
# popup), quindi il job vive in st.session_state e ogni rerun si riaggancia ad esso.
# Il thread NON deve usare st.*: comunica solo attraverso l'oggetto Job.
class Job:
    def __init__(self):
        self.log = RunLog()
        self.status_label = "Elaborazione in corso..."
        self.done = False
        self.final = {"label": "", "state": "complete"}
        self.result = None            # dict {paper, type, results, llm} se c'e' qualcosa da mostrare
        self.question = None          # domanda pendente per l'utente (popup)
        self.answer = None
        self.answer_event = threading.Event()

    def ask_unavailable(self, llm: LLM, reason: str, invalid: bool = False) -> str:
        """Chiamata dal thread quando il provider resta non disponibile (o il modello non esiste):
        blocca fino alla scelta dell'utente. Ritorna "retry", "switch" (llm.model gia' cambiato) o "abort"."""
        self.log("[~] Recupero l'elenco dei modelli del provider...")
        models = list_models(llm)
        ordered, suggested = rank_alternatives(llm.model, models)
        preferred = PROVIDERS[llm.provider].get("fallback_model")   # alternativa voluta per questo provider
        if preferred and preferred != llm.model and (preferred in models or not models):
            suggested = preferred
            ordered = [preferred] + [m for m in ordered if m != preferred]
        elif not suggested:
            suggested = None
        self.answer = None
        self.answer_event.clear()
        self.question = {"provider": llm.label, "model": llm.model, "reason": reason, "invalid": invalid,
                         "models": ordered, "suggested": suggested}
        self.log("[?] In attesa di una scelta dell'utente...")
        answered = self.answer_event.wait(timeout=QUESTION_TIMEOUT)
        self.question = None
        decision, new_model = self.answer if answered and self.answer else ("abort", None)
        if decision == "switch" and new_model:
            self.log(f"[~] Cambio modello su richiesta dell'utente: {llm.model} -> {new_model}")
            llm.model = new_model
        elif decision == "retry":
            self.log(f"[~] Nuovi tentativi con {llm.model} su richiesta dell'utente.")
        else:
            decision = "abort"
            self.log("[!] Analisi interrotta dall'utente." if answered else "[!] Nessuna risposta: analisi interrotta.")
        return decision


def start_job(target, *args) -> Job:
    """Avvia target(job, *args) in un thread. Ritorna None se un'altra analisi e' gia' in corso sul server."""
    lock = _pipeline_lock()
    if not lock.acquire(blocking=False):
        return None
    job = Job()

    def _run():
        try:
            target(job, *args)
        except AnalysisAborted:
            job.final = {"label": "Analisi interrotta", "state": "error"}
        except Exception as e:
            job.log(f"[!] Errore inatteso: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            job.final = {"label": f"Errore inatteso: {e}", "state": "error"}
        finally:
            job.done = True
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return job


def _hooked(llm: LLM, job: Job) -> LLM:
    return replace(llm, log=job.log, on_unavailable=job.ask_unavailable)


def _final_state(results: dict) -> dict:
    failed = [k for k, r in results.items() if r["data"] is None]
    if not failed:
        return {"label": "Analisi completata!", "state": "complete"}
    if len(failed) < len(results):
        return {"label": f"Analisi completata con errori ({', '.join(failed)})", "state": "error"}
    return {"label": "Analisi non riuscita", "state": "error"}


def job_full_analysis(job: Job, llm: LLM, paper_meta: dict, analysis_type: str, iterations: int, max_figures: int):
    log, llm = job.log, _hooked(llm, job)
    want_pdf, want_latex = analysis_type in ("both", "pdf"), analysis_type in ("both", "latex")
    results = {}
    try:
        log(f"Paper: {paper_meta['Title']} (arXiv:{paper_meta['ArXiv_ID']})\nLLM: {llm.label} / {llm.model}")
        paper = download_sources(paper_meta, want_pdf, want_latex, log)
        missing = [name for name, wanted, path in (("PDF", want_pdf, paper["Local_PDF_Path"]),
                                                   ("LaTeX", want_latex, paper["Local_LaTeX_Path"]))
                   if wanted and path == "N/A"]
        if missing:
            msg = f"Impossibile scaricare: {', '.join(missing)}. Scegli un altro paper o un'altra sorgente."
            log(f"[!] {msg}")
            job.final = {"label": msg, "state": "error"}
            return
        for key, wanted, kind, path_key, status in (("PDF", want_pdf, "pdf", "Local_PDF_Path", "📄 Pipeline sul PDF in corso..."),
                                                    ("LaTeX", want_latex, "latex", "Local_LaTeX_Path", "📦 Pipeline sul LaTeX in corso...")):
            if wanted:
                job.status_label = status
                log("=" * 70 + f"\nPIPELINE ON {key.upper()}\n" + "=" * 70)
                results[key] = run_source(llm, paper[path_key], kind, iterations, max_figures, log)
        job.final = _final_state(results)
    finally:
        if results:  # anche se interrotta: mostra cio' che e' stato completato
            job.result = {"paper": paper, "type": analysis_type, "results": results, "llm": replace(llm, log=lambda m: None, on_unavailable=None)}


def job_reanalyze(job: Job, analysis: dict):
    """Rilancia solo l'analisi finale delle sorgenti fallite, riusando il testo arricchito (niente vision/PySR)."""
    llm = _hooked(analysis["llm"], job)
    results = dict(analysis["results"])
    try:
        for key, res in analysis["results"].items():
            if res["data"] is None and res.get("enriched_text"):
                job.status_label = f"🔁 Nuova analisi finale {key}..."
                outcome = analyze_with_llm(llm, res["enriched_text"], job.log)
                results[key] = {**res, **{k: outcome[k] for k in ("model", "data", "raw", "attempts")},
                                "error": outcome["error"] if outcome["data"] is None else None}
        job.final = _final_state(results)
    finally:
        job.result = {**analysis, "results": results, "llm": replace(llm, log=lambda m: None, on_unavailable=None)}


# ==========================================
# INTERFACCIA UTENTE
# ==========================================
def _format_value(value) -> str:
    if value is None or value == "" or value == []:
        return "_Not specified_"
    if isinstance(value, list):
        return "\n".join(f"- {v}" for v in value)
    if isinstance(value, dict):
        return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"
    return str(value)


def display_source_result(label: str, res: dict):
    if res is None:
        st.info(f"Sorgente {label} non analizzata.")
        return
    st.caption(f"File: `{os.path.basename(res['source'])}` · modello: `{res.get('model') or 'n/d'}` · "
               f"testo arricchito: {res['text_chars']:,} caratteri · figure trovate: {len(res['figures'])}, "
               f"analizzate: {len(res['analyzed_figures'])}")
    for w in res.get("warnings", []):
        st.warning(w)

    data = res.get("data")
    if data is None:
        st.error(f"Analisi {label} non riuscita: {res.get('error') or 'motivo sconosciuto'}")
        if res.get("raw"):
            with st.expander("Risposta grezza del modello"):
                st.code(res["raw"], language=None)
        if res.get("enriched_text"):
            st.button("🔁 Riprova solo l'analisi finale", key=f"reanalyze_{label}", on_click=request_reanalysis,
                      help="Riusa il testo già arricchito (figure e PySR già fatti): rifà una sola chiamata al modello.")
        return

    for field in FIELDS:
        st.subheader(FIELD_LABELS[field])
        value = data.get(field)
        if field == "visual_analysis_findings":
            if value:
                st.warning(_format_value(value))
            else:
                st.write("_Nessuna formula ricavata dai grafici._")
        else:
            st.markdown(_format_value(value))

    extra = {k: v for k, v in data.items() if k not in FIELDS}
    if extra:
        with st.expander("Campi extra restituiti dal modello"):
            st.json(extra)
    if res.get("advanced_blocks"):
        with st.expander("Sezione ADVANCED PLOT ANALYSIS inviata al modello"):
            st.code("\n\n".join(res["advanced_blocks"]), language=None)
    with st.expander("JSON completo"):
        st.json(data)
        st.download_button(f"⬇️ Scarica JSON {label}", json.dumps(data, indent=2, ensure_ascii=False),
                           file_name=f"analysis_{label.lower()}.json", mime="application/json", key=f"dl_{label}")


def display_comparison(results: dict):
    pdf_res, latex_res = results.get("PDF"), results.get("LaTeX")
    missing = [name for name, r in (("PDF", pdf_res), ("LaTeX", latex_res)) if r is None or r.get("data") is None]
    if missing:
        st.warning(f"Confronto non disponibile: l'analisi {' e '.join(missing)} non ha prodotto un JSON valido. "
                   "Servono entrambe le analisi riuscite per confrontarle.")
        return
    st.subheader("⚖️ Confronto diretto PDF vs LaTeX, campo per campo")
    for field in FIELDS:
        st.markdown(f"**{FIELD_LABELS[field]}**")
        c1, c2 = st.columns(2)
        c1.caption("Da PDF")
        c1.markdown(_format_value(pdf_res["data"].get(field)))
        c2.caption("Da LaTeX")
        c2.markdown(_format_value(latex_res["data"].get(field)))
        st.divider()


def display_figures(results: dict):
    shown = False
    for label, res in results.items():
        if not res or not res.get("figures"):
            continue
        analyzed = set(res["analyzed_figures"])
        existing = [p for p in res["figures"] if os.path.exists(p)]
        if not existing:
            continue
        shown = True
        st.subheader(f"Figure da {label} ({len(existing)})")
        cols = st.columns(3)
        for i, path in enumerate(existing):
            tag = "analizzata" if path in analyzed else "non analizzata (limite max figure)"
            try:
                cols[i % 3].image(path, caption=f"{os.path.basename(path)} — {tag}", width="stretch")
            except Exception:
                cols[i % 3].caption(f"{os.path.basename(path)} (anteprima non disponibile)")
    if not shown:
        st.info("Nessuna figura estratta.")


st.title("📄 Scientific Paper Analyzer")
st.markdown("Analisi automatizzata di paper scientifici (PDF e LaTeX) da **INSPIRE HEP** con LLM, "
            "Computer Vision e Symbolic Regression — stessa pipeline di `demo_pipeline_standalone.ipynb`.")

config = load_config()
saved_keys = {name: ("" if config.get(spec["config_key"], "") == KEY_PLACEHOLDER else config.get(spec["config_key"], ""))
              for name, spec in PROVIDERS.items()}

# Tutti gli input stanno in form: toccare un widget fuori da un form fa ripartire lo script
# e interromperebbe un'analisi in corso.

# ---- Passo 1: ricerca (solo metadati, nessun download)
with st.form("search_form"):
    st.subheader("🔍 1. Cerca paper su INSPIRE HEP")
    col1, col2 = st.columns(2)
    with col1:
        q_arxiv = st.text_input("arXiv ID (es. 0907.3624):")
        q_title = st.text_input("Parole nel titolo:", help="Basta anche una sola parola: l'ID non serve, lo trova la ricerca.")
        q_author = st.text_input("Autore:")
    with col2:
        q_raw = st.text_input("Query INSPIRE libera (ignora i campi a sinistra):", placeholder="es. t flux tube and date > 2015")
        q_min_citations = st.number_input("Citazioni minime:", min_value=0, value=0)
        q_limit = st.number_input("Risultati da mostrare:", min_value=1, max_value=25, value=10)
        q_sort = st.selectbox("Ordinamento:", [None, "mostrecent", "mostcited"],
                              format_func=lambda x: {None: "Rilevanza", "mostrecent": "Più recenti", "mostcited": "Più citati"}[x])
    st.caption("Se lasci vuoti tutti i campi di ricerca vengono mostrati i paper più citati (`topcite 100+`).")
    search_button = st.form_submit_button("🔍 Cerca", type="primary")

if search_button:
    if q_raw.strip():
        search_query = q_raw.strip()
    else:
        parts = []
        if q_arxiv.strip():
            parts.append(f"arxiv:{q_arxiv.strip()}")
        if q_title.strip():
            parts.append(f"t {q_title.strip()}")
        if q_author.strip():
            parts.append(f"a {q_author.strip()}")
        search_query = " and ".join(parts) if parts else "topcite 100+"
    try:
        with st.spinner("Ricerca su INSPIRE HEP..."):
            found = search_inspire_hep(search_query, int(q_limit), int(q_min_citations), q_sort)
        st.session_state["search"] = {"query": search_query, "papers": found}
    except RuntimeError as e:
        st.session_state.pop("search", None)
        st.error(f"Errore INSPIRE HEP: {e}")

search = st.session_state.get("search")
run_button = False
if search is not None and not search["papers"]:
    st.warning(f"Nessun paper trovato per `{search['query']}`: cambia query o abbassa le citazioni minime.")

# ---- Passo 2: scelta del paper e configurazione dell'analisi
if search and search["papers"]:
    papers_found = search["papers"]

    def _paper_label(i: int) -> str:
        p_ = papers_found[i]
        arxiv = f"arXiv:{p_['ArXiv_ID']}" if p_["ArXiv_ID"] != "N/A" else "senza arXiv, non analizzabile"
        return f"{p_['Title']} — {p_['Authors']} ({p_['Year'] or 's.d.'}) · {p_['Citations']} citazioni · {arxiv}"

    with st.form("analysis_form"):
        st.subheader(f"📄 2. Scegli il paper da analizzare ({len(papers_found)} risultati per `{search['query']}`)")
        chosen = st.radio("Paper:", list(range(len(papers_found))), format_func=_paper_label, label_visibility="collapsed")

        st.subheader("🤖 3. Modello LLM")
        provider_names = list(PROVIDERS)
        default_provider = config.get("LLM_PROVIDER") if config.get("LLM_PROVIDER") in PROVIDERS else "openrouter"
        m1, m2 = st.columns(2)
        provider = m1.selectbox("Provider:", provider_names, index=provider_names.index(default_provider),
                                format_func=lambda n: PROVIDERS[n]["label"])
        model_input = m2.text_input("Modello (vuoto = default del provider):",
                                    placeholder=" · ".join(f"{spec['label']}: {spec['model']}" for spec in PROVIDERS.values()),
                                    help="Deve supportare input immagine. Default = modello standard del provider.")
        with st.expander("🔑 API key", expanded=not any(saved_keys.values())):
            st.caption("Serve solo la chiave del provider selezionato sopra.")
            key_inputs = {}
            for name, spec in PROVIDERS.items():
                key_inputs[name] = st.text_input(f"{spec['label']} — API key:", value=saved_keys[name], type="password",
                                                 help=f"Si crea su {spec['key_help']}", key=f"key_{name}")
            remember_key = st.checkbox("Ricorda le chiavi", value=any(saved_keys.values()),
                                       help=f"Salva le chiavi in chiaro in {CONFIG_FILE} (escluso da git).")
        st.warning("💳 **Costi e privacy.** Con Claude e OpenAI le chiamate sono **a pagamento** sulla tua chiave. "
                   "Gemini ha un free tier (5 richieste/minuto per modello): l'app distanzia le chiamate e attende "
                   "quando il limite scatta, quindi l'analisi è più lenta ma arriva in fondo; con billing attivo "
                   "le chiamate Gemini sono a pagamento. "
                   "La pipeline fa molte chiamate con immagini: un controllo per ogni pagina del PDF che cita figure, "
                   "più la digitalizzazione di ogni figura analizzata, per ciascuna sorgente scelta (PDF e LaTeX). "
                   "Riduci le figure massime e scegli una sola sorgente per contenere il costo. "
                   "Testo del paper e immagini vengono inviati al provider scelto.")

        analysis_type = st.radio("Sorgente:", ["both", "pdf", "latex"], horizontal=True,
                                 format_func=lambda x: {"both": "Entrambi (PDF + LaTeX)", "pdf": "Solo PDF", "latex": "Solo LaTeX"}[x])
        with st.expander("🔧 Parametri analisi"):
            max_figures = st.slider("Max figure da analizzare per sorgente (0 = tutte):", 0, 15, 5)
            pysr_iterations = st.slider("Iterazioni symbolic regression (PySR) per curva:", 5, 60, 20)
        run_button = st.form_submit_button("🚀 Avvia analisi", type="primary")


@st.dialog("⚠️ Modello non disponibile", dismissible=False)
def unavailable_dialog(job: Job):
    q = job.question
    if q is None:  # la risposta e' gia' stata data (rerun in corso)
        st.rerun()
    if q["invalid"] and q["reason"].startswith("HTTP 429"):
        st.write(f"La **quota** del modello `{q['model']}` su {q['provider']} è **esaurita** (es. il limite "
                 "giornaliero del free tier). L'analisi è in pausa: ogni modello ha la sua quota, "
                 "scegli un altro modello per continuare.")
    elif q["invalid"]:
        st.write(f"Il modello `{q['model']}` **non è utilizzabile** su {q['provider']}. L'analisi è in pausa: "
                 "scegli un altro modello.")
    else:
        st.write(f"**{q['provider']}** non risponde sul modello `{q['model']}` anche dopo vari tentativi: "
                 "il server è sovraccarico. L'analisi è in pausa.")
    st.caption(q["reason"][:300])

    if q["models"]:
        options = q["models"]
        picked = st.selectbox("Modello alternativo (elenco reale del provider):", options,
                              index=options.index(q["suggested"]) if q["suggested"] in options else 0)
    else:
        picked = q["suggested"] or ""
        st.caption("Elenco dei modelli non disponibile: scrivi l'ID del modello da usare.")
    custom = st.text_input("Oppure scrivi un ID modello a mano:", value="" if q["models"] else (picked or ""))
    target = custom.strip() or picked
    if not q["invalid"]:
        st.warning(f"Un altro modello può essere **meno potente**, quindi i risultati possono essere meno accurati. "
                   f"Se invece resti su `{q['model']}` con il server sovraccarico rischi di **sprecare chiamate** "
                   "(a pagamento) e che l'analisi **non vada a buon fine**.")
    choice = None
    cols = st.columns(2 if q["invalid"] else 3)
    if cols[0].button(f"Passa a {target}" if target else "Passa al modello scelto", type="primary",
                      disabled=not target, use_container_width=True):
        choice = ("switch", target)
    if not q["invalid"] and cols[1].button(f"Riprova con {q['model']}", use_container_width=True):
        choice = ("retry", None)
    if cols[-1].button("Interrompi analisi", use_container_width=True):
        choice = ("abort", None)
    if choice:
        job.answer = choice
        job.question = None
        job.answer_event.set()
        st.rerun()


def render_job(job: Job):
    """Mostra l'avanzamento del job e, se serve, il popup. Alla fine travasa il risultato in session_state."""
    with st.status(job.status_label, expanded=True) as status:
        placeholder, shown, label = st.empty(), -1, None
        while not job.done and job.question is None:
            if len(job.log.lines) != shown:
                shown = len(job.log.lines)
                placeholder.code("\n".join(job.log.lines[-300:]), language=None)
            if job.status_label != label:
                label = job.status_label
                status.update(label=label)
            time.sleep(0.5)
        placeholder.code("\n".join(job.log.lines[-300:]), language=None)
        if job.done:
            status.update(label=job.final["label"], state=job.final["state"], expanded=False)
        else:
            status.update(label="⏸ In pausa: serve una tua scelta", state="running")
    if not job.done:
        unavailable_dialog(job)
        st.stop()
    if job.result:
        st.session_state["analysis"] = {**job.result, "log": list(job.log.lines)}
    st.session_state.pop("job", None)


if run_button:
    paper_meta = papers_found[chosen]
    llm = LLM(provider, key_inputs[provider].strip(), model_input.strip() or PROVIDERS[provider]["model"])
    if "job" in st.session_state:
        st.warning("⏳ C'è già un'analisi in corso: attendi che finisca.")
    elif not llm.api_key or llm.api_key == KEY_PLACEHOLDER:
        st.error(f"⚠️ Inserisci una API key {PROVIDERS[provider]['label']} valida.")
    elif paper_meta["ArXiv_ID"] == "N/A":
        st.error("⚠️ Questo paper non ha un arXiv ID: PDF e LaTeX non sono scaricabili. Scegline un altro.")
    else:
        new_config = {k: v for k, v in config.items() if k not in {spec["config_key"] for spec in PROVIDERS.values()}}
        new_config["LLM_PROVIDER"] = provider
        if remember_key:
            for name, spec in PROVIDERS.items():
                if key_inputs[name].strip():
                    new_config[spec["config_key"]] = key_inputs[name].strip()
        save_config(new_config)

        new_job = start_job(job_full_analysis, llm, paper_meta, analysis_type, pysr_iterations, max_figures)
        if new_job is None:
            st.warning("⏳ C'è già un'analisi in corso su questo server: attendi che finisca e riprova.")
        else:
            st.session_state.pop("analysis", None)
            st.session_state["job"] = new_job

if "job" in st.session_state:
    render_job(st.session_state["job"])


def request_reanalysis():
    """Callback del pulsante 'Riprova solo l'analisi finale'."""
    new_job = start_job(job_reanalyze, st.session_state["analysis"])
    if new_job is None:
        st.session_state["notice"] = "⏳ C'è già un'analisi in corso su questo server: attendi che finisca e riprova."
    else:
        st.session_state["job"] = new_job


if st.session_state.get("notice"):
    st.warning(st.session_state.pop("notice"))

analysis = st.session_state.get("analysis")
if analysis:
    paper, results = analysis["paper"], analysis["results"]
    st.markdown("---")
    st.header("📊 Risultati")
    st.markdown(f"**{paper['Title']}** · arXiv:`{paper['ArXiv_ID']}` · citazioni: {paper['Citations']} · INSPIRE id `{paper['Paper_ID']}`")

    tab_names = [k for k in ("PDF", "LaTeX") if k in results]
    if analysis["type"] == "both":
        tab_names.append("Confronto")
    tab_names += ["Figure", "Log"]
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    for key in ("PDF", "LaTeX"):
        if key in tabs:
            with tabs[key]:
                display_source_result(key, results.get(key))
    if "Confronto" in tabs:
        with tabs["Confronto"]:
            display_comparison(results)
    with tabs["Figure"]:
        display_figures(results)
    with tabs["Log"]:
        st.code("\n".join(analysis["log"]), language=None)
