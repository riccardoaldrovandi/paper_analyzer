#!/bin/bash
#SBATCH --job-name=nli_judges
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nli_judges_%j.out
#SBATCH --error=logs/nli_judges_%j.err

# Runs the groundedness judges on gold/claims.jsonl, submitted from the repo root:
#   sbatch scripts/nli/run_judges.sbatch
# Optional overrides (--export=ALL,VAR=...):
#   JUDGES   space-separated judge names (default: the four small ones;
#            add bespoke-minicheck-7b after `WITH_BESPOKE=1 download_models.sh`)
#   RECORDS  export-dataset JSONL with the paper texts
#            (default logs_cineca/qwen38/qwen38_gold_ctx128k_s0.jsonl)
#   OUT_DIR  predictions directory (default logs/nli_predictions)
# Each judge runs in its own process: one that fails does not stop the others.

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/nli/common.sh
source "$NLI_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet

JUDGES="${JUDGES:-minicheck-flan-t5 minicheck-deberta minicheck-roberta deberta-mnli}"
RECORDS="${RECORDS:-logs_cineca/qwen38/qwen38_gold_ctx128k_s0.jsonl}"
OUT_DIR="${OUT_DIR:-logs/nli_predictions}"
[ -f "$RECORDS" ] || { echo "[!] $RECORDS not found (rsync it from the PC)"; exit 1; }
[ -f gold/claims.jsonl ] || { echo "[!] gold/claims.jsonl not found"; exit 1; }

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
torch.ones(1, device='cuda'); print(f'[+] torch {torch.__version__}, CUDA works')
" || exit 1

FAILED=""
for J in $JUDGES; do
    echo; echo "=== $J ($(date +%T))"
    python scripts/nli/benchmark_judges.py --judge "$J" --claims gold/claims.jsonl \
        --records "$RECORDS" --out-dir "$OUT_DIR" || FAILED="$FAILED $J"
done

echo; echo "=== Scores"
python scripts/nli/score_judges.py --claims gold/claims.jsonl --predictions "$OUT_DIR" \
    --json "$OUT_DIR/summary.json"
[ -z "$FAILED" ] || { echo "[!] Judges that failed:$FAILED"; exit 1; }
