#!/bin/bash
#SBATCH --job-name=groundedness
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/groundedness_%j.out
#SBATCH --error=logs/groundedness_%j.err

# Scores every claim of logs/groundedness/claims.jsonl with Bespoke-MiniCheck-7B.
# Submitted from the repo root:  sbatch scripts/groundedness/run_groundedness.sbatch
# Overrides: CLAIMS, RECORDS_DIR (default logs_cineca/qwen38), OUT.
# Resumable: submit it again after a timeout and it continues where it stopped.

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs logs/groundedness
source scripts/nli/common.sh
source "$NLI_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes have no internet

CLAIMS="${CLAIMS:-logs/groundedness/claims.jsonl}"
RECORDS_DIR="${RECORDS_DIR:-logs_cineca/qwen38}"
OUT="${OUT:-logs/groundedness/scores.jsonl}"
[ -f "$CLAIMS" ] || { echo "[!] $CLAIMS not found (rsync it from the PC)"; exit 1; }
[ -d "$RECORDS_DIR" ] || { echo "[!] $RECORDS_DIR not found (rsync the run JSONL files)"; exit 1; }

nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
torch.ones(1, device='cuda'); print(f'[+] torch {torch.__version__}, CUDA works')
" || exit 1

python scripts/groundedness/judge_claims.py --claims "$CLAIMS" --records-dir "$RECORDS_DIR" --out "$OUT"
