#!/bin/bash
#SBATCH --job-name=diagnose_text
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnose_text_%j.out
#SBATCH --error=logs/diagnose_text_%j.err

# Why does a text model stop after a few tokens? (scripts/diagnose_text_model.py)
#   sbatch --export=ALL,TEXT_MODEL=velvet-14b scripts/diagnose_text_model.sbatch

module purge
module load gcc/12.2.0
source "$WORK/venv_vllm/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TEXT_MODEL="${TEXT_MODEL:-velvet-14b}"
INPUTS="${INPUTS:-training_data_qwen_v7.jsonl}"
LOCAL_MODEL_PATH="${MODELS_DIR:-$WORK/models}/$TEXT_MODEL" \
    python scripts/diagnose_text_model.py --inputs "$INPUTS"
