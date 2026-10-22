#!/bin/bash
#SBATCH --job-name=judge_controls
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/judge_controls_%j.out
#SBATCH --error=logs/judge_controls_%j.err

# One GPU, Bespoke-MiniCheck-7B: (1) the control claims of docs/METRICS.md 7.4
# (negative control + perturbed claims), (2) SciFact dev with oracle abstracts.
# Needs on the cluster: logs/groundedness/controls_claims.jsonl (made on the PC by
# make_controls.py), logs_cineca/qwen38/*.jsonl, and $WORK/scifact/data
# (scripts/scifact/prepare_scifact.sh on a login node).
#   sbatch scripts/groundedness/run_controls.sbatch
# Resumable (the controls skip what is scored; SciFact takes ~5 min and reruns).
# Overrides: CONTROLS, RECORDS_DIR, SCIFACT_DIR, SKIP_SCIFACT=1.

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs logs/groundedness
source scripts/nli/common.sh
source "$NLI_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CONTROLS="${CONTROLS:-logs/groundedness/controls_claims.jsonl}"
RECORDS_DIR="${RECORDS_DIR:-logs_cineca/qwen38}"
SCIFACT_DIR="${SCIFACT_DIR:-$WORK_DIR/scifact/data}"
[ -f "$CONTROLS" ] || { echo "[!] $CONTROLS not found"; exit 1; }
python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
torch.ones(1, device='cuda'); print(f'[+] torch {torch.__version__}, CUDA works')
" || exit 1

echo "[~] 1/2 control claims"
python scripts/groundedness/judge_claims.py --claims "$CONTROLS" --records-dir "$RECORDS_DIR" \
    --out logs/groundedness/controls_scores.jsonl
STATUS=$?

if [ "${SKIP_SCIFACT:-0}" != 1 ]; then
    echo "[~] 2/2 SciFact dev (oracle abstracts)"
    python scripts/scifact/run_scifact.py --data "$SCIFACT_DIR" --out logs/scifact_bespoke.json || STATUS=1
fi
exit $STATUS
