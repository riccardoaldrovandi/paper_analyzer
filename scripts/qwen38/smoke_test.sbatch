#!/bin/bash
#SBATCH --job-name=qwen38_smoke
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# Half a node, as in run_dual_model.sbatch (default memory got vLLM OOM-killed).
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:45:00
#SBATCH --output=logs/qwen38_smoke_%j.out
#SBATCH --error=logs/qwen38_smoke_%j.err

# Smoke test of Qwen3.8-27B on vLLM, submitted from the repo root:
#   sbatch scripts/qwen38/smoke_test.sbatch
# Optional overrides: MAX_MODEL_LEN (default 32768), MAX_NUM_SEQS (default 32),
# GPU_MEM_UTIL (default 0.85), ENFORCE_EAGER=1 (skip
# torch.compile/CUDA graphs if they fail), QWEN38_RUNTIME=venv|container.
#
# Qwen3.8-27B in BF16 is ~52 GB of weights: one 64 GB A100 would leave almost
# nothing for the KV cache, so ONE server spans both GPUs of this job with
# tensor parallelism (~26 GB of weights per GPU). Unlike run_dual_model.sbatch
# there is no text/vision split: the same model reads text and images.

module purge
module load gcc/12.2.0   # C compiler for Triton/torch.compile (venv runtime)
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/qwen38/common.sh
source scripts/vllm_server.sh   # wait_for_vision_server / stop_vision_server
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet

[ -f "$QWEN38_MODEL_DIR/model.safetensors.index.json" ] || { echo "[!] Model not found in $QWEN38_MODEL_DIR: run scripts/qwen38/download_model.sh"; exit 1; }
echo "[~] Runtime: $(qwen38_runtime) | model: $QWEN38_MODEL_DIR | GPUs: '$CUDA_VISIBLE_DEVICES'"
qwen38_start_and_wait "${MAX_MODEL_LEN:-32768}" || exit 1

"${QWEN38_RUN[@]}" python3 scripts/qwen38/smoke_test.py --port "$PORT" \
    --figure scripts/probe_figures/flux_tube.png \
    --out "logs/qwen38_smoke_${SLURM_JOB_ID}.json"
STATUS=$?

echo; echo "[~] Server facts from $SERVER_LOG:"
grep -iE "weights|KV cache|maximum concurrency|attention backend|error" "$SERVER_LOG" | head -20
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
exit $STATUS
