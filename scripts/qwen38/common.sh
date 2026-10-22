# Shared settings for the Qwen3.8 scripts. Sourced (not executed) by
# setup_env.sh, download_model.sh and smoke_test.sbatch.
#
# Qwen3.8-27B (model_type qwen3_5, hybrid Gated DeltaNet + attention) needs
# vLLM >= 0.17, while $WORK/venv_vllm is pinned to vLLM 0.7.3: this runtime
# lives apart from it, so the old baseline environment stays untouched.
# vLLM 0.19.1 is the newest release whose PyPI wheel is still manylinux_2_31
# (0.20+ needs glibc 2.35) and whose torch (2.10.0) still has a CUDA 12.x build,
# which the CUDA 12.2 driver of the A100 nodes can run.

WORK_DIR="${WORK:?WORK is not set (it is on Leonardo)}"
MODELS_DIR="${MODELS_DIR:-$WORK_DIR/models}"
QWEN38_MODEL_DIR="${QWEN38_MODEL_DIR:-$MODELS_DIR/qwen3.8-27b}"
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
QWEN38_VENV="${QWEN38_VENV:-$WORK_DIR/venv_vllm_qwen38}"
QWEN38_SIF="${QWEN38_SIF:-$WORK_DIR/containers/vllm-openai_v${VLLM_VERSION}.sif}"
APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"

# Compile caches off $HOME (small quota).
export VLLM_CACHE_ROOT="$WORK_DIR/.cache/vllm"
export TRITON_CACHE_DIR="$WORK_DIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/.cache/torchinductor"
# FlashInfer's sampler JIT-compiles CUDA code with nvcc on first use: the
# PyTorch sampler needs no compiler and is fine for this throughput.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Which runtime setup_env.sh produced: "venv" (glibc >= 2.31) or "container"
# (Apptainer image, for older glibc). Force one with QWEN38_RUNTIME.
qwen38_runtime() {
    if [ -n "${QWEN38_RUNTIME:-}" ]; then echo "$QWEN38_RUNTIME"
    elif [ -x "$QWEN38_VENV/bin/vllm" ]; then echo venv
    elif [ -f "$QWEN38_SIF" ]; then echo container
    else echo none
    fi
}

# Sets QWEN38_RUN, the command prefix that runs a program inside the runtime:
#   "${QWEN38_RUN[@]}" vllm serve ...    "${QWEN38_RUN[@]}" python3 script.py
# An array (not a function) so it also works after setsid.
qwen38_init_run() {
    case "$(qwen38_runtime)" in
        venv)      QWEN38_RUN=(env "PATH=$QWEN38_VENV/bin:$PATH" "VIRTUAL_ENV=$QWEN38_VENV") ;;
        container) QWEN38_RUN=("$APPTAINER_BIN" exec --nv --bind /leonardo_work "$QWEN38_SIF") ;;
        *) echo "[!] No Qwen3.8 runtime found: run scripts/qwen38/setup_env.sh on a login node first." >&2; return 1 ;;
    esac
}

# start_qwen38_server [max_model_len] -- inside a 2-GPU SLURM job: one vLLM
# server for Qwen3.8-27B spanning both GPUs (tensor parallelism: ~52 GB of BF16
# weights do not leave room for a KV cache on one 64 GB A100). Sets PORT,
# SERVER_LOG and VISION_PID; pair it with wait_for_vision_server /
# stop_vision_server from scripts/vllm_server.sh. The model is served under its
# path (no --served-model-name), which is the "model" the pipeline sends.
#   --max-num-seqs 32: the default 256 made the sampler warm-up (256 dummy
#   requests x a 248K-token vocabulary) OOM both GPUs at 0.90 (job 61441931).
start_qwen38_server() {
    local max_len="${1:-${MAX_MODEL_LEN:-32768}}" extra=""
    [ "${ENFORCE_EAGER:-0}" = 1 ] && extra="--enforce-eager"
    PORT="${PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
    SERVER_LOG="logs/qwen38_server_${SLURM_JOB_ID}.log"
    echo "[~] Starting Qwen3.8 vLLM server (TP=2, max_model_len=$max_len, port $PORT), log: $SERVER_LOG"
    # setsid: own process group, so stop_vision_server also kills the TP workers.
    setsid "${QWEN38_RUN[@]}" vllm serve "$QWEN38_MODEL_DIR" \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
        --max-num-seqs "${MAX_NUM_SEQS:-32}" \
        --reasoning-parser qwen3 \
        --limit-mm-per-prompt '{"image": 2, "video": 0}' \
        --port "$PORT" $extra \
        > "$SERVER_LOG" 2>&1 &
    VISION_PID=$!
}

# Environment for the pipeline (venv_vllm, vLLM 0.7.3) to use the Qwen3.8
# server for both text and figures, over HTTP. Thinking off by default.
qwen38_pipeline_env() {
    export LOCAL_MODEL_PATH="$QWEN38_MODEL_DIR" LOCAL_VISION_MODEL_PATH="$QWEN38_MODEL_DIR"
    export LOCAL_TEXT_SERVER_URL="http://localhost:$PORT/v1" LOCAL_VISION_SERVER_URL="http://localhost:$PORT/v1"
    export LOCAL_VISION_GUIDED_JSON=1 LOCAL_VISION_JSON_MODE=response_format LOCAL_VISION_TIMEOUT=600
    export LOCAL_VISION_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
    export LOCAL_ENABLE_THINKING="${LOCAL_ENABLE_THINKING:-0}"
}

# Fails in seconds, not after a 4-minute weights load, if the runtime cannot
# use the GPUs (e.g. CUDA runtime too new for the driver) or got only one.
qwen38_cuda_check() {
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    "${QWEN38_RUN[@]}" python3 -c "
import torch
n = torch.cuda.device_count()
print(f'[~] torch {torch.__version__} CUDA {torch.version.cuda}, visible GPUs: {n}')
assert n >= 2, 'need 2 GPUs for tensor parallelism'
for i in range(n): torch.ones(1, device=f'cuda:{i}')
print('[+] CUDA works on all GPUs')
" || { echo "[!] CUDA check failed -- see above. Nothing else was started."; return 1; }
}

# qwen38_start_and_wait <max_model_len>: CUDA check + server start + wait.
# Installs the trap that always stops the server when the job ends.
qwen38_start_and_wait() {
    qwen38_cuda_check || return 1
    start_qwen38_server "$1"
    trap stop_vision_server EXIT
    if ! wait_for_vision_server "$PORT"; then
        echo "[!] Server did not come up. Last lines of $SERVER_LOG:"
        tail -40 "$SERVER_LOG"
        return 1
    fi
}
