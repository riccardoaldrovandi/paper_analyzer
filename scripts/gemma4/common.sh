# Shared settings for the Gemma 4 scripts. Sourced (not executed).
#
# Gemma 4 31B (dense, text + images, Apache 2.0) runs in the SAME runtime as
# Qwen3.8: the vLLM 0.19.1 Apptainer image built by scripts/qwen38/setup_env.sh.
# Everything that is not model-specific comes from scripts/qwen38/common.sh.

source "$(dirname "${BASH_SOURCE[0]}")/../qwen38/common.sh"

GEMMA4_MODEL_DIR="${GEMMA4_MODEL_DIR:-$MODELS_DIR/gemma4-31b}"

# start_gemma4_server [max_model_len] -- like start_qwen38_server: ONE server
# over the job's 2 GPUs (31B in BF16 is ~62 GB of weights: TP=2). Sets PORT,
# SERVER_LOG and VISION_PID. Differences from Qwen3.8:
#   --reasoning-parser gemma4   splits the thought channel from the answer
#   --limit-mm-per-prompt       image only (the flag's JSON form is the one that worked for Qwen)
# Gemma 4 needs far more KV cache than Qwen3.8 (TP=2, 85%: only 50k tokens), so a
# 128k context needs GEMMA4_TP=4 (a 4-GPU job). Overrides: GEMMA4_TP, GEMMA4_EXTRA_ARGS (extra `vllm serve` flags, e.g. a chat template).
start_gemma4_server() {
    local max_len="${1:-${MAX_MODEL_LEN:-32768}}" extra=""
    [ "${ENFORCE_EAGER:-0}" = 1 ] && extra="--enforce-eager"
    PORT="${PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
    SERVER_LOG="logs/gemma4_server_${SLURM_JOB_ID}.log"
    echo "[~] Starting Gemma 4 vLLM server (TP=${GEMMA4_TP:-2}, max_model_len=$max_len, port $PORT), log: $SERVER_LOG"
    setsid "${QWEN38_RUN[@]}" vllm serve "$GEMMA4_MODEL_DIR" \
        --tensor-parallel-size "${GEMMA4_TP:-2}" \
        --dtype bfloat16 \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
        --max-num-seqs "${MAX_NUM_SEQS:-32}" \
        --reasoning-parser gemma4 \
        --limit-mm-per-prompt '{"image": 2}' \
        --port "$PORT" $extra ${GEMMA4_EXTRA_ARGS:-} \
        > "$SERVER_LOG" 2>&1 &
    VISION_PID=$!
}

gemma4_start_and_wait() {
    qwen38_cuda_check || return 1
    start_gemma4_server "$1"
    trap stop_vision_server EXIT
    if ! wait_for_vision_server "$PORT"; then
        echo "[!] Server did not come up. Last lines of $SERVER_LOG:"
        tail -60 "$SERVER_LOG"
        return 1
    fi
}

# Environment for the pipeline (venv_vllm, vLLM 0.7.3) to use the Gemma 4 server
# for both text and figures, over HTTP. Thinking off by default.
gemma4_pipeline_env() {
    export LOCAL_MODEL_PATH="$GEMMA4_MODEL_DIR" LOCAL_VISION_MODEL_PATH="$GEMMA4_MODEL_DIR"
    export LOCAL_TEXT_SERVER_URL="http://localhost:$PORT/v1" LOCAL_VISION_SERVER_URL="http://localhost:$PORT/v1"
    export LOCAL_VISION_GUIDED_JSON=1 LOCAL_VISION_JSON_MODE=response_format LOCAL_VISION_TIMEOUT=600
    export LOCAL_VISION_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
    export LOCAL_ENABLE_THINKING="${LOCAL_ENABLE_THINKING:-0}"
}
