import json
import os

from src.config import ADVANCED_HEADER
from src.utils.helpers import repair_json_backslashes, strip_json_fences, extract_json_object

_llm = None

# Overridable via the LOCAL_MODEL_PATH environment variable; default = where
# Qwen2.5-14B-Instruct was downloaded on CINECA ($MODELS_DIR, i.e.
# $WORK/models unless set, + qwen2.5-14b).
# The model comparison (Qwen vs Llama vs Mistral) just points this elsewhere.
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.environ.get("WORK", "."), "models"))
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", os.path.join(MODELS_DIR, "qwen2.5-14b"))
# Optional label of the run configuration (e.g. "ctx128k", "think"), appended
# to MODEL_NAME so `evaluate` treats each configuration of the same model as a
# separate provider instead of pooling them as repeated runs. Empty = no tag.
RUN_TAG = os.environ.get("LOCAL_RUN_TAG", "").strip()
# Derived from the model directory so dataset records are never mislabeled
# when a different text model is loaded.
MODEL_NAME = f"{os.path.basename(MODEL_PATH.rstrip('/'))}{f' [{RUN_TAG}]' if RUN_TAG else ''} (local/vLLM)"
# If set (e.g. http://localhost:20001/v1), the text model is NOT loaded in this
# process: requests go to a running vLLM OpenAI-compatible server instead.
# Needed for models the in-process vLLM 0.7.3 can't load (Qwen3.8, served from
# the vLLM 0.19.1 container, see scripts/qwen38/); unset = in-process as before.
TEXT_SERVER_URL = os.environ.get("LOCAL_TEXT_SERVER_URL", "").rstrip("/")

# Per-model tuning knobs: Mistral-Nemo recommends temperature ~0.3, and models
# with a long default context (Mistral-Nemo: 128k) need a cap to fit the A100.
TEMPERATURE = float(os.environ.get("LOCAL_TEMPERATURE", "0.2"))
MAX_MODEL_LEN = int(os.environ.get("LOCAL_MAX_MODEL_LEN", "32768"))
MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MAX_NEW_TOKENS", "4096"))
# Fixed sampling seed, so a rerun with the same inputs and settings gives the
# same answer and model comparisons can be reproduced.
SEED = int(os.environ.get("LOCAL_SEED", "0"))
# Constrain the answer to ANSWER_SCHEMA (vLLM guided decoding, xgrammar). Test
# job 59806387: valid JSON 3/3 for both Qwen and Mistral-Nemo (Mistral free:
# 2/3), same or better L1/L2, +13-33% generation time. Set to 0 to disable.
GUIDED_JSON = os.environ.get("LOCAL_GUIDED_JSON", "1") != "0"

# Optional sampling overrides, all unset by default (= the settings every
# earlier comparison used). The Qwen3.8 "think" configuration sets them to the
# model card's thinking-mode values.
_OPTIONAL_SAMPLING = {
    name: float(os.environ[var]) if name != "top_k" else int(os.environ[var])
    for name, var in [("top_p", "LOCAL_TOP_P"), ("top_k", "LOCAL_TOP_K"), ("presence_penalty", "LOCAL_PRESENCE_PENALTY")]
    if os.environ.get(var, "").strip()
}
# Hybrid thinking models (Qwen3.x): "1" = think before answering, "0" = answer
# directly, unset = the chat template's default (and no flag sent at all, as
# for every model before Qwen3.8). HTTP mode only.
ENABLE_THINKING = {"1": True, "0": False}.get(os.environ.get("LOCAL_ENABLE_THINKING", "").strip())

# Extra fields merged into every HTTP request body (JSON), e.g. Gemma 4 needs
# {"skip_special_tokens": false} for the server to split its thought channel
# from the answer. Unset = nothing added.
EXTRA_BODY = json.loads(os.environ.get("LOCAL_EXTRA_BODY", "") or "{}")

# Saved in every export-dataset record, so each label can be traced back to
# the exact settings that produced it.
GENERATION_SETTINGS = {
    "temperature": TEMPERATURE,
    "max_model_len": MAX_MODEL_LEN,
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "guided_json": GUIDED_JSON,
    "repetition_penalty": float(os.environ.get("LOCAL_REPETITION_PENALTY", "1.15")),
    "frequency_penalty": float(os.environ.get("LOCAL_FREQUENCY_PENALTY", "0.3")),
    **_OPTIONAL_SAMPLING,
    # Only recorded when used, so records of the earlier models are unchanged.
    **({"text_server": True} if TEXT_SERVER_URL else {}),
    **({"enable_thinking": ENABLE_THINKING} if ENABLE_THINKING is not None else {}),
    **({"run_tag": RUN_TAG} if RUN_TAG else {}),
    **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
}

USER_PREFIX = "--- PAPER CONTENT ---\n"
# Room left for the chat template's special tokens and role headers.
_TEMPLATE_MARGIN = 256
# Share of the paper body kept from the start when truncating; the rest comes
# from the end, where conclusions, limitations and future work usually are.
_HEAD_SHARE = 0.7


def _get_llm():
    """
    Loads the model only once per process. export-dataset processes many
    papers in the same run: reloading ~15-28 GB of weights on every call
    would be pointlessly slow.
    """
    global _llm
    if _llm is None:
        from vllm import LLM
        print(f"[~] Loading local model from {MODEL_PATH} (first call only, max_model_len={MAX_MODEL_LEN})...")
        _llm = LLM(model=MODEL_PATH, dtype="bfloat16", max_model_len=MAX_MODEL_LEN)
    return _llm


class _ServerTokenizer:
    """
    encode/decode through the server's /tokenize and /detokenize endpoints, so
    fit_to_context() counts tokens with exactly the served model's tokenizer
    without loading it (or vLLM) in this process.
    """

    def _post(self, endpoint: str, payload: dict) -> dict:
        import requests
        base = TEXT_SERVER_URL[:-3] if TEXT_SERVER_URL.endswith("/v1") else TEXT_SERVER_URL
        response = requests.post(f"{base}/{endpoint}", json={"model": MODEL_PATH, **payload}, timeout=120)
        response.raise_for_status()
        return response.json()

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        return self._post("tokenize", {"prompt": text, "add_special_tokens": add_special_tokens})["tokens"]

    def decode(self, ids: list) -> str:
        return self._post("detokenize", {"tokens": list(ids)})["prompt"]


def _get_tokenizer():
    return _ServerTokenizer() if TEXT_SERVER_URL else _get_llm().get_tokenizer()


def fit_to_context(text_content: str, tokenizer=None, reserve_tokens: int = 0, system_prompt: str = None) -> tuple[str, bool]:
    """
    Makes the paper text fit the model's context window. About half of the
    downloaded papers (46/98 on 2026-10-09) are longer than 32k tokens, and
    vLLM refuses a prompt longer than max_model_len. If the text is too long,
    the paper body is cut in the middle (keeping its start and its end) while
    the ADVANCED PLOT ANALYSIS block is always kept whole. Returns the text
    actually sent to the model and whether it was truncated; export-dataset
    stores that text, so each training example matches what the teacher saw.

    reserve_tokens: room kept for what is sent besides the text (the images of
    the single multimodal call); system_prompt: the one actually sent, if not
    SYSTEM_PROMPT.
    """
    from src.config import SYSTEM_PROMPT

    tokenizer = tokenizer or _get_tokenizer()
    system_prompt = system_prompt or SYSTEM_PROMPT

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    budget = MAX_MODEL_LEN - MAX_NEW_TOKENS - count(system_prompt) - count(USER_PREFIX) - _TEMPLATE_MARGIN - reserve_tokens
    if count(text_content) <= budget:
        return text_content, False

    cut = text_content.find(ADVANCED_HEADER)
    body, advanced = (text_content[:cut], text_content[cut:]) if cut != -1 else (text_content, "")
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    advanced_ids = tokenizer.encode(advanced, add_special_tokens=False) if advanced else []
    if len(advanced_ids) > budget // 2:
        # Never seen in practice: a figure block this large would leave no room
        # for the paper, so it gets cut too (from its end).
        advanced_ids = advanced_ids[:budget // 2]
        advanced = tokenizer.decode(advanced_ids)

    body_budget = budget - len(advanced_ids) - 64  # 64 tokens for the omission note
    for _ in range(3):
        head_n = int(body_budget * _HEAD_SHARE)
        tail_n = body_budget - head_n
        head = tokenizer.decode(body_ids[:head_n])
        tail = tokenizer.decode(body_ids[-tail_n:]) if tail_n > 0 else ""
        omitted = max(0, len(body) - len(head) - len(tail))
        note = f"\n\n[... {omitted} characters of the paper omitted here to fit the model's context window ...]\n\n"
        fitted = head + note + tail.rstrip() + ("\n\n" + advanced if advanced else "")
        excess = count(fitted) - budget
        if excess <= 0:
            break
        # Decoding and re-encoding can shift token boundaries slightly.
        body_budget -= excess + 32

    print(f"[!] Paper text too long for the {MAX_MODEL_LEN}-token context: kept the start and end of the body "
          f"({omitted} characters omitted){' and the whole figure analysis block' if advanced else ''}.")
    return fitted, True


def _chat_via_server(messages: list, sampling_kwargs: dict, schema: dict = None) -> str:
    """
    One chat completion against TEXT_SERVER_URL with the same sampling settings
    as the in-process path. Returns the answer text only: with a reasoning
    parser on the server, a thinking model's reasoning comes back in a separate
    field and never reaches the JSON parsing.
    """
    import requests
    payload = {"model": MODEL_PATH, "messages": messages, **sampling_kwargs}
    if ENABLE_THINKING is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
    payload.update(EXTRA_BODY)
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    # A 128k-token prompt plus a long thinking phase can take several minutes.
    response = requests.post(f"{TEXT_SERVER_URL}/chat/completions", json=payload, timeout=1800)
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if reasoning:
        print(f"[~] Model reasoning: {len(reasoning)} characters (not part of the answer).")
    # "length" = cut by max_tokens (reasoning included): tells a truncated
    # answer apart from one the model ended itself (e.g. after a loop).
    finish = choice.get("finish_reason")
    tokens = (body.get("usage") or {}).get("completion_tokens")
    print(f"[~] Generation ended: finish_reason={finish}, {tokens} completion token(s) (max {sampling_kwargs.get('max_tokens')}).")
    if finish == "length":
        print("[!] Answer cut by max_tokens: it is likely truncated, invalid JSON.")
    return message.get("content") or ""


def _image_part(path: str) -> dict:
    import base64
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(path, "rb") as f:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"}}


def analyze_with_local(text_content: str, images: list = None, system_prompt: str = None, schema: dict = None) -> tuple[str, str]:
    """
    The final analysis call. images (paths) and system_prompt are used by the
    single multimodal call (export-dataset --mode single): the images go before
    the text, as in the figure digitization requests. HTTP server mode only.
    schema replaces ANSWER_SCHEMA for guided decoding (the figure verification
    pass of export-dataset asks for the "figures" list only).
    """
    from src.config import SYSTEM_PROMPT

    system_prompt = system_prompt or SYSTEM_PROMPT
    # No-op when the caller already passed the text through fit_to_context().
    text_content, _ = fit_to_context(text_content, system_prompt=system_prompt)
    user = f"{USER_PREFIX}{text_content}"
    if images:
        if not TEXT_SERVER_URL:
            raise RuntimeError("Images in the final call need a vLLM server (LOCAL_TEXT_SERVER_URL).")
        user = [_image_part(p) for p in images] + [{"type": "text", "text": user}]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    # repetition_penalty/frequency_penalty added after the 2026-10-06 bug: with
    # no penalty at all, the model occasionally spirals into repeating a single
    # token (e.g. "0") until max_tokens cuts it off mid-JSON (see
    # project-local-model-repetition-bug memory for the root cause). Kept the
    # same for every model so the comparison only varies the model itself.
    sampling_kwargs = dict(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=GENERATION_SETTINGS["repetition_penalty"],
        frequency_penalty=GENERATION_SETTINGS["frequency_penalty"],
        seed=SEED,
        **_OPTIONAL_SAMPLING,
    )

    print(f"[~] Generating with local {MODEL_NAME} (temperature={TEMPERATURE}, seed={SEED}, guided JSON={GUIDED_JSON}"
          f"{f', {len(images)} image(s)' if images else ''}"
          f"{f', thinking={ENABLE_THINKING}' if ENABLE_THINKING is not None else ''}{', via server' if TEXT_SERVER_URL else ''})...")
    if TEXT_SERVER_URL:
        raw_text = None
        if GUIDED_JSON:
            from src.llm.schemas import ANSWER_SCHEMA
            try:
                raw_text = _chat_via_server(messages, sampling_kwargs, schema or ANSWER_SCHEMA)
            except Exception as e:
                # Same policy as in-process: a guided failure must not cost the paper.
                print(f"[!] Guided JSON generation failed ({type(e).__name__}: {e}); retrying without the schema constraint.")
        if raw_text is None:
            raw_text = _chat_via_server(messages, sampling_kwargs)
        return MODEL_NAME, repair_json_backslashes(extract_json_object(strip_json_fences(raw_text)))

    from vllm import SamplingParams
    llm = _get_llm()
    outputs = None
    if GUIDED_JSON:
        from vllm.sampling_params import GuidedDecodingParams
        from src.llm.schemas import ANSWER_SCHEMA
        try:
            guided = GuidedDecodingParams(json=ANSWER_SCHEMA)
            outputs = llm.chat(messages, SamplingParams(**sampling_kwargs, guided_decoding=guided))
        except Exception as e:
            # A guided-decoding failure must not cost the paper: retry unconstrained
            # (the salvage logic in main.py still covers a broken findings field).
            print(f"[!] Guided JSON generation failed ({type(e).__name__}: {e}); retrying without the schema constraint.")
    if outputs is None:
        outputs = llm.chat(messages, SamplingParams(**sampling_kwargs))
    raw_text = outputs[0].outputs[0].text
    # Same cleanup for every model, so the comparison isn't decided by which
    # one happens to wrap its JSON in ```json fences or a leading sentence.
    raw_text = repair_json_backslashes(extract_json_object(strip_json_fences(raw_text)))

    return MODEL_NAME, raw_text
