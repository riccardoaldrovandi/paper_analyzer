import os

from src.config import ADVANCED_HEADER
from src.utils.helpers import repair_json_backslashes, strip_json_fences, extract_json_object

_llm = None

# Overridable via the LOCAL_MODEL_PATH environment variable; default = where
# Qwen2.5-14B-Instruct was downloaded on CINECA ($WORK/models/qwen2.5-14b).
# The model comparison (Qwen vs Llama vs Mistral) just points this elsewhere.
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/leonardo_work/IscrC_MultLLM/models/qwen2.5-14b")
# Derived from the model directory so dataset records are never mislabeled
# when a different text model is loaded.
MODEL_NAME = f"{os.path.basename(MODEL_PATH.rstrip('/'))} (local/vLLM)"

# Per-model tuning knobs: Mistral-Nemo recommends temperature ~0.3, and models
# with a long default context (Mistral-Nemo: 128k) need a cap to fit the A100.
TEMPERATURE = float(os.environ.get("LOCAL_TEMPERATURE", "0.2"))
MAX_MODEL_LEN = int(os.environ.get("LOCAL_MAX_MODEL_LEN", "32768"))
MAX_NEW_TOKENS = 4096
# Fixed sampling seed, so a rerun with the same inputs and settings gives the
# same answer and model comparisons can be reproduced.
SEED = int(os.environ.get("LOCAL_SEED", "0"))
# Constrain the answer to ANSWER_SCHEMA (vLLM guided decoding, xgrammar). Test
# job 59806387: valid JSON 3/3 for both Qwen and Mistral-Nemo (Mistral free:
# 2/3), same or better L1/L2, +13-33% generation time. Set to 0 to disable.
GUIDED_JSON = os.environ.get("LOCAL_GUIDED_JSON", "1") != "0"

# Saved in every export-dataset record, so each label can be traced back to
# the exact settings that produced it.
GENERATION_SETTINGS = {
    "temperature": TEMPERATURE,
    "max_model_len": MAX_MODEL_LEN,
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "guided_json": GUIDED_JSON,
    "repetition_penalty": 1.15,
    "frequency_penalty": 0.3,
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


def fit_to_context(text_content: str, tokenizer=None) -> tuple[str, bool]:
    """
    Makes the paper text fit the model's context window. About half of the
    downloaded papers (46/98 on 2026-10-09) are longer than 32k tokens, and
    vLLM refuses a prompt longer than max_model_len. If the text is too long,
    the paper body is cut in the middle (keeping its start and its end) while
    the ADVANCED PLOT ANALYSIS block is always kept whole. Returns the text
    actually sent to the model and whether it was truncated; export-dataset
    stores that text, so each training example matches what the teacher saw.
    """
    from src.config import SYSTEM_PROMPT

    tokenizer = tokenizer or _get_llm().get_tokenizer()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    budget = MAX_MODEL_LEN - MAX_NEW_TOKENS - count(SYSTEM_PROMPT) - count(USER_PREFIX) - _TEMPLATE_MARGIN
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


def analyze_with_local(text_content: str) -> tuple[str, str]:
    from vllm import SamplingParams
    from src.config import SYSTEM_PROMPT

    llm = _get_llm()
    # No-op when the caller already passed the text through fit_to_context().
    text_content, _ = fit_to_context(text_content)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_PREFIX}{text_content}"},
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
    )

    print(f"[~] Generating with local {MODEL_NAME} (temperature={TEMPERATURE}, seed={SEED}, guided JSON={GUIDED_JSON})...")
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
