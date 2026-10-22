#!/bin/bash
#SBATCH --job-name=qwen38_crop
# No --account here: set SBATCH_ACCOUNT in your shell, as for run_dual_model.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/qwen38_crop_%j.out
#SBATCH --error=logs/qwen38_crop_%j.err

# Crops the figures of PDF_DIR (default gold_pdfs) into FIGURES_DIR (default
# gold_figures) with Qwen3.8 as page filter + plot locator, replacing the
# Gemini step that is unavailable on CINECA. Every model of a comparison then
# reads the same figures via --figures-dir. Submitted by submit_all.sh, or:
#   sbatch scripts/qwen38/crop_figures.sbatch

module purge
module load gcc/12.2.0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/qwen38/common.sh
source scripts/vllm_server.sh   # wait_for_vision_server / stop_vision_server
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PDF_DIR="${PDF_DIR:-gold_pdfs}"
FIGURES_DIR="${FIGURES_DIR:-gold_figures}"
qwen38_start_and_wait 32768 || exit 1

# The pipeline's own Python (PyMuPDF etc.) talks to the server over HTTP.
source "$WORK/venv_vllm/bin/activate"
qwen38_pipeline_env
LOCAL_FIGURE_CROP=1 python scripts/qwen38/crop_figures.py --pdf-dir "$PDF_DIR" --out "$FIGURES_DIR"
