# Helpers for running a local vLLM vision server inside a SLURM job.
# Sourced (not executed) by run_dual_model.sbatch and probe_vision_models.sbatch,
# so both start, wait for and stop the server in exactly the same way.

MODELS_DIR="${MODELS_DIR:-$WORK/models}"

# Model-specific `vllm serve` flags, keyed by the directory name under $MODELS_DIR.
vision_server_args() {
    case "$1" in
        qwen2.5-vl-7b)        echo "--max-model-len 8192" ;;
        # Llama-3-8B context is 8192: 16384 crashes at config time.
        llava-next-llama3-8b) echo "--max-model-len 8192" ;;
        # The tokenizer's own chat template is plain Llama-3 format, not the
        # "User:<image>...<end_of_utterance>" format Idefics3 was trained on.
        idefics3-8b-llama3)   echo "--max-model-len 8192 --chat-template $SLURM_SUBMIT_DIR/scripts/chat_templates/idefics3.jinja" ;;
        # Mistral-native weights; native-resolution images need a longer context.
        pixtral-12b)          echo "--tokenizer-mode mistral --config-format mistral --load-format mistral --max-model-len 16384" ;;
        *) echo "[!] Unknown vision model '$1': add it to vision_server_args in scripts/vllm_server.sh" >&2; return 1 ;;
    esac
}

# start_vision_server <model_dir_name> <gpu_index> <port>
# Sets VISION_PID. setsid puts the server in its own process group, so
# stop_vision_server can kill vLLM's engine child process too (killing only
# the parent left it resident until the job's cgroup teardown -> oom_kill).
start_vision_server() {
    local model="$1" gpu="$2" port="$3" extra
    extra=$(vision_server_args "$model") || return 1
    echo "[~] Starting vision server: $model on GPU $gpu, port $port ($extra)"
    CUDA_VISIBLE_DEVICES="$gpu" setsid vllm serve "$MODELS_DIR/$model" \
        --dtype bfloat16 --port "$port" $extra \
        > "logs/vision_server_${model}_${SLURM_JOB_ID}.log" 2>&1 &
    VISION_PID=$!
}

# wait_for_vision_server <port> -- startup takes 125-430s and varies a lot with
# filesystem load, so poll for up to 15 min (SERVER_WAIT_STEPS x 5 s: Mistral
# Small 4 sets 30 min), but give up immediately if the
# server process died.
wait_for_vision_server() {
    local port="$1"
    local steps="${SERVER_WAIT_STEPS:-180}"
    for i in $(seq 1 "$steps"); do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "[+] Vision server ready after $((i * 5))s."
            return 0
        fi
        if ! kill -0 "$VISION_PID" 2>/dev/null; then
            echo "[!] Vision server process exited during startup -- see its log."
            return 1
        fi
        sleep 5
    done
    echo "[!] Vision server not ready after $((steps * 5))s."
    return 1
}

stop_vision_server() {
    [ -z "$VISION_PID" ] && return 0
    echo "[~] Stopping vision server (process group $VISION_PID)..."
    kill -TERM -- "-$VISION_PID" 2>/dev/null || pkill -TERM -P "$VISION_PID" 2>/dev/null
    for i in $(seq 1 30); do
        kill -0 "$VISION_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$VISION_PID" 2>/dev/null
    wait "$VISION_PID" 2>/dev/null
    VISION_PID=""
}
