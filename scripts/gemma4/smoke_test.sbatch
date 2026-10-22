#!/bin/bash
#SBATCH --job-name=gemma4_smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:45:00
#SBATCH --output=logs/gemma4_smoke_%j.out
#SBATCH --error=logs/gemma4_smoke_%j.err

# Smoke test of Gemma 4 31B on vLLM 0.19.1 (container), from the repo root:
#   sbatch scripts/gemma4/smoke_test.sbatch
# Same four checks as Qwen3.8 (text no-think, text think with the reasoning
# parser, image, guided JSON). Overrides: MAX_MODEL_LEN (32768), MAX_NUM_SEQS
# (32), GPU_MEM_UTIL (0.85), ENFORCE_EAGER=1, GEMMA4_EXTRA_ARGS.

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/gemma4/common.sh
source scripts/vllm_server.sh
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

[ -f "$GEMMA4_MODEL_DIR/model.safetensors.index.json" ] || { echo "[!] Model not found in $GEMMA4_MODEL_DIR: run scripts/gemma4/download_model.sh"; exit 1; }
echo "[~] Runtime: $(qwen38_runtime) | model: $GEMMA4_MODEL_DIR | GPUs: '$CUDA_VISIBLE_DEVICES'"
gemma4_start_and_wait "${MAX_MODEL_LEN:-32768}" || exit 1

# Gemma 4 sampling (model card): temperature 1.0, top_p 0.95, top_k 64. Thinking
# is switched with chat_template_kwargs.enable_thinking; skip_special_tokens
# False is what lets the reasoning parser split thought from answer.
"${QWEN38_RUN[@]}" python3 scripts/qwen38/smoke_test.py --port "$PORT" \
    --figure scripts/probe_figures/flux_tube.png \
    --out "logs/gemma4_smoke_${SLURM_JOB_ID}.json" \
    --nothink-json '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": false}}' \
    --think-json '{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "skip_special_tokens": false, "chat_template_kwargs": {"enable_thinking": true}}'
STATUS=$?

echo; echo "[~] Server facts from $SERVER_LOG:"
grep -iE "weights|KV cache|maximum concurrency|attention backend|error" "$SERVER_LOG" | head -20
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
exit $STATUS
