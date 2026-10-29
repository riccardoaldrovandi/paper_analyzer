# Shared settings for the Mistral Small 4 scripts. Sourced (not executed).
#
# Mistral Small 4 (mistralai/Mistral-Small-4-119B-2603, Apache 2.0): 119B MoE
# (128 experts, 4 active + 1 shared, ~6.5B active per token), MLA attention,
# Pixtral vision encoder, 256k context, reasoning switchable per request with
# "reasoning_effort" ("none" / "high"). Weights are FP8 (~121 GB).
#
# Runs in the same vLLM 0.19.1 runtime as Qwen3.8 and Gemma 4
# (scripts/qwen38/setup_env.sh). That release loads this architecture only from
# the Mistral-native format (params.json + consolidated-*.safetensors, mapped
# to MistralLarge3ForCausalLM + Pixtral by vllm/transformers_utils/configs/
# mistral.py); the HF format of the same repo (model_type "mistral4") needs
# transformers 5.3-dev, which vLLM 0.19.1 excludes. Hence --*-format mistral.
#
# A100 specifics (the model card targets H100/H200):
#   - FP8: no native FP8 on Ampere; vLLM runs the weights as W8A16 (Marlin
#     kernels). Same memory, slower than on Hopper.
#   - MLA: the card's --attention-backend FLASH_ATTN_MLA is Hopper-only; on
#     A100 vLLM picks its own MLA backend (Triton), so no flag is passed.
#   - Memory: ~121 GB of weights do not fit 2 x 64 GB: TP=4 (a whole node).
#     The MLA KV cache is small (~23 KB/token), so a 128k context is cheap.
#
# The official FP8 weights produce GARBAGE on A100 with vLLM 0.19.1 (jobs
# 62432709, 62433938 eager, 62433943 MLA disabled, 2026-10-27: digits and
# zero-width spaces even for "capital of France?"). The only thing those runs
# shared is the FP8-on-Marlin path, so the default variant is a 4-bit AWQ
# (W4A16, group 32, compressed-tensors) of the same model, which runs on the
# int4 Marlin kernels: cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit, ~71 GB,
# Mistral format too. Experts and the large attention projections are 4-bit;
# shared experts, q_a/kv_a projections, vision tower and lm_head stay BF16.
# Not the official weights: report results as "Mistral Small 4 (AWQ 4-bit)".
#   MISTRAL4_VARIANT=awq (default, TP=2) | fp8 (official, TP=4: kept to re-test
#   after a vLLM update)

source "$(dirname "${BASH_SOURCE[0]}")/../qwen38/common.sh"

MISTRAL4_VARIANT="${MISTRAL4_VARIANT:-awq}"
case "$MISTRAL4_VARIANT" in
    awq) MISTRAL4_REPO=cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit
         MISTRAL4_MODEL_DIR="${MISTRAL4_MODEL_DIR:-$MODELS_DIR/mistral-small-4-119b-awq}"
         MISTRAL4_TP="${MISTRAL4_TP:-2}" ;;
    fp8) MISTRAL4_REPO=mistralai/Mistral-Small-4-119B-2603
         MISTRAL4_MODEL_DIR="${MISTRAL4_MODEL_DIR:-$MODELS_DIR/mistral-small-4-119b}"
         MISTRAL4_TP="${MISTRAL4_TP:-4}" ;;
    *) echo "[!] MISTRAL4_VARIANT must be awq or fp8, not '$MISTRAL4_VARIANT'" >&2; return 1 ;;
esac
MISTRAL4_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No internet on compute nodes: the usage-stats thread only adds a traceback to the log.
export VLLM_NO_USAGE_STATS=1

# Newer mistral_common, only if the container's is too old (see
# mistral4_mistral_common_check): installed once into this directory and put
# in front of the container's own packages. Container runtime only.
MISTRAL4_PYDEPS="${MISTRAL4_PYDEPS:-$WORK_DIR/pydeps_mistral4}"
if [ -d "$MISTRAL4_PYDEPS" ]; then
    export APPTAINERENV_PYTHONPATH="$MISTRAL4_PYDEPS${APPTAINERENV_PYTHONPATH:+:$APPTAINERENV_PYTHONPATH}"
fi

# start_mistral4_server [max_model_len] -- like start_gemma4_server. Sets PORT,
# SERVER_LOG and VISION_PID. Overrides: MISTRAL4_TP (awq 2, fp8 4), GPU_MEM_UTIL (0.85),
# MAX_NUM_SEQS (32), MISTRAL4_MAX_IMAGES (2), MISTRAL4_EXTRA_ARGS.
#
# Two fixes found in the first smoke test (job 62431278, 2026-10-27):
#   - patch_vllm_mla.py: vLLM 0.19.1 crashes ("unsupported `a` scalar_type") on
#     any prefill that reuses cached tokens with FP8 weights on A100; the patched
#     module is bind-mounted over the container's one (container runtime only).
#   - disable_any_whitespace: with guided JSON the model filled the 512-token
#     budget with newlines (133 lines for a 3-field object). xgrammar must be
#     named explicitly: the option is refused with backend "auto".
start_mistral4_server() {
    local max_len="${1:-${MAX_MODEL_LEN:-131072}}" extra="" bind
    [ "${ENFORCE_EAGER:-0}" = 1 ] && extra="--enforce-eager"
    if [ "$(qwen38_runtime)" != container ]; then
        echo "[!] The MLA patch is applied by bind-mounting into the container: runtime '$(qwen38_runtime)' is not supported." >&2
        return 1
    fi
    bind=$("${QWEN38_RUN[@]}" python3 "$MISTRAL4_SCRIPTS/patch_vllm_mla.py" "$WORK_DIR/.cache/mistral4_patch") || return 1
    export APPTAINER_BIND="$bind${APPTAINER_BIND:+,$APPTAINER_BIND}" SINGULARITY_BIND="$bind${SINGULARITY_BIND:+,$SINGULARITY_BIND}"
    # Inside the container the original path must now show the patched line.
    "${QWEN38_RUN[@]}" grep -q "_kv_b_proj_w_dtype.is_floating_point" "${bind#*:}" \
        || { echo "[!] MLA patch not visible inside the container (bind $bind)." >&2; return 1; }
    echo "[+] vLLM MLA patch active: $bind"
    PORT="${PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
    SERVER_LOG="logs/mistral4_server_${SLURM_JOB_ID}.log"
    echo "[~] Starting Mistral Small 4 vLLM server (TP=$MISTRAL4_TP, max_model_len=$max_len, port $PORT), log: $SERVER_LOG"
    setsid "${QWEN38_RUN[@]}" vllm serve "$MISTRAL4_MODEL_DIR" \
        --tokenizer-mode mistral --config-format mistral --load-format mistral \
        --tensor-parallel-size "$MISTRAL4_TP" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
        --max-num-seqs "${MAX_NUM_SEQS:-32}" \
        --max-num-batched-tokens 16384 \
        --reasoning-parser mistral \
        --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
        --limit-mm-per-prompt "{\"image\": ${MISTRAL4_MAX_IMAGES:-${MAX_IMAGES:-2}}}" \
        --port "$PORT" $extra ${MISTRAL4_EXTRA_ARGS:-} \
        > "$SERVER_LOG" 2>&1 &
    VISION_PID=$!
}

# The model card asks for mistral_common >= 1.11.0; vLLM 0.19.1 only requires
# >= 1.10.0, so the container may ship an older one. Checked before the 10+
# minute weights load, so a mismatch costs seconds.
mistral4_mistral_common_check() {
    "${QWEN38_RUN[@]}" python3 -c "
import mistral_common, vllm
v = tuple(int(x) for x in mistral_common.__version__.split('.')[:2])
print(f'[~] vllm {vllm.__version__}, mistral_common {mistral_common.__version__}')
assert v >= (1, 11), 'mistral_common < 1.11.0: Mistral Small 4 needs >= 1.11.0 (fix: header of scripts/mistral4/smoke_test.sbatch)'
" || return 1
}

# mistral4_start_and_wait <max_model_len>: checks + server start + wait. The
# weights load takes longer than the 15 min other models need: 30 min allowed.
mistral4_start_and_wait() {
    qwen38_cuda_check || return 1
    mistral4_mistral_common_check || return 1
    start_mistral4_server "$1"
    trap stop_vision_server EXIT
    if ! SERVER_WAIT_STEPS=360 wait_for_vision_server "$PORT"; then
        echo "[!] Server did not come up. Last lines of $SERVER_LOG:"
        tail -60 "$SERVER_LOG"
        return 1
    fi
}

# Environment for the pipeline (venv_vllm) to use the Mistral server for both
# text and figures, over HTTP. Reasoning is a request field, not a chat
# template switch: LOCAL_ENABLE_THINKING stays unset (no chat_template_kwargs)
# and MISTRAL4_REASONING (none | high) sets reasoning_effort for the final
# answer. Figures are always digitized with reasoning off, as for the others.
# skip_special_tokens true (the default): with false, multi-byte characters
# split across byte tokens came out as U+FFFD ("ρ��", 20 of them in the test line),
# in the answers of the 2026-10-27 matrix too. scripts/mistral4/utf8_check.py
# (run by every matrix job, job 62571411) found the reasoning still separated
# from the answer with true, so the flag is no longer needed for [THINK].
mistral4_pipeline_env() {
    export LOCAL_MODEL_PATH="$MISTRAL4_MODEL_DIR" LOCAL_VISION_MODEL_PATH="$MISTRAL4_MODEL_DIR"
    export LOCAL_TEXT_SERVER_URL="http://localhost:$PORT/v1" LOCAL_VISION_SERVER_URL="http://localhost:$PORT/v1"
    export LOCAL_VISION_GUIDED_JSON=1 LOCAL_VISION_JSON_MODE=response_format LOCAL_VISION_TIMEOUT=600
    export LOCAL_VISION_EXTRA_BODY='{"reasoning_effort": "none"}'
    export LOCAL_EXTRA_BODY="{\"reasoning_effort\": \"${MISTRAL4_REASONING:-none}\"}"
    unset LOCAL_ENABLE_THINKING
}
