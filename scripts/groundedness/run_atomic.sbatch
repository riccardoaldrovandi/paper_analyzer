#!/bin/bash
#SBATCH --job-name=atomic_grounded
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:30:00
#SBATCH --output=logs/atomic_grounded_%j.out
#SBATCH --error=logs/atomic_grounded_%j.err

# Splits the sentence-level claims into atomic claims with Qwen3.8 (stage A, 2 GPUs,
# TP=2) and scores the atomic claims with Bespoke-MiniCheck-7B (stage B, 1 GPU).
# Submitted from the repo root:  sbatch scripts/groundedness/run_atomic.sbatch
# Overrides (--export=ALL,VAR=...):
#   RUNS         space-separated run names (default: Qwen3.8 think + Qwen2.5 baseline, 3 seeds each);
#                'all' = every run found in CLAIMS (e.g. the 9 Gemma 4 runs)
#   CLAIMS       sentence-level claims (default logs/groundedness/claims.jsonl)
#   RECORDS_DIR  folder with the <run>.jsonl files (default logs_cineca/qwen38)
#   OUTDIR       folder for atoms.jsonl, atomic_claims.jsonl, atomic_scores.jsonl (default logs/groundedness)
# Gemma 4 example:
#   sbatch --export=ALL,RUNS=all,CLAIMS=logs/groundedness_gemma4/claims.jsonl,RECORDS_DIR=logs_cineca/gemma4,OUTDIR=logs/groundedness_gemma4 \
#       scripts/groundedness/run_atomic.sbatch
# Resumable: both stages skip what is already written, so after a timeout just
# submit it again.

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
OUTDIR="${OUTDIR:-logs/groundedness}"
mkdir -p logs "$OUTDIR"

RUNS="${RUNS:-qwen38_gold_think_s0 qwen38_gold_think_s1 qwen38_gold_think_s2 qwen_baseline_gold_s0 qwen_baseline_gold_s1 qwen_baseline_gold_s2}"
CLAIMS="${CLAIMS:-logs/groundedness/claims.jsonl}"
RECORDS_DIR="${RECORDS_DIR:-logs_cineca/qwen38}"
ATOMS="$OUTDIR/atoms.jsonl"
ATOMIC="$OUTDIR/atomic_claims.jsonl"
SCORES="$OUTDIR/atomic_scores.jsonl"
RUNS_ARG=(--runs $RUNS)
[ "$RUNS" = all ] && RUNS_ARG=()
[ -f "$CLAIMS" ] || { echo "[!] $CLAIMS not found (rsync it from the PC)"; exit 1; }
[ -d "$RECORDS_DIR" ] || { echo "[!] $RECORDS_DIR not found (rsync the run JSONL files)"; exit 1; }

# ---- Stage A: decomposition with Qwen3.8 (same container/venv as scripts/qwen38)
source scripts/qwen38/common.sh
source scripts/vllm_server.sh   # wait_for_vision_server / stop_vision_server
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet
[ -f "$QWEN38_MODEL_DIR/model.safetensors.index.json" ] || { echo "[!] Model not found in $QWEN38_MODEL_DIR"; exit 1; }

echo "[~] Stage A: decomposition of the claims of: $RUNS"
qwen38_start_and_wait 8192 || exit 1   # sentences are short: a small context is enough
"${QWEN38_RUN[@]}" python3 scripts/groundedness/decompose_claims.py --claims "$CLAIMS" \
    --url "http://localhost:$PORT/v1" --atoms "$ATOMS" --out "$ATOMIC" "${RUNS_ARG[@]}"
STATUS=$?
stop_vision_server
trap - EXIT
[ $STATUS -eq 0 ] || { echo "[!] Decomposition failed (status $STATUS)"; exit $STATUS; }

# ---- Stage B: judge the atomic claims with Bespoke-MiniCheck-7B on one GPU
echo "[~] Stage B: judging $(wc -l < "$ATOMIC") atomic claims"
source scripts/nli/common.sh
source "$NLI_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
torch.ones(1, device='cuda'); print(f'[+] torch {torch.__version__}, CUDA works')
" || exit 1
python scripts/groundedness/judge_claims.py --claims "$ATOMIC" --records-dir "$RECORDS_DIR" --out "$SCORES"
