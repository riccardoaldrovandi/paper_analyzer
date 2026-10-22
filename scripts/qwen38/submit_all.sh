#!/bin/bash
# Submits everything needed to evaluate Qwen3.8-27B on the gold set, with the
# right dependencies. Run on a LOGIN node, from the repo root:
#   bash scripts/qwen38/submit_all.sh
#
#   1. crop_figures.sbatch  Qwen3.8 crops the figures of PDF_DIR into FIGURES_DIR
#                           (skipped if FIGURES_DIR already has figures)
#   2. run_eval.sbatch      Qwen3.8: vision probe + CONFIGS x SEEDS
#   3. run_dual_model.sbatch baseline(s) on the SAME papers and figures, one job
#                           per seed: the first digitizes the figures, the
#                           others reuse its checkpoints (same vision input)
# Overrides: PDF_DIR, FIGURES_DIR, SET, SEEDS, CONFIGS, BASELINES (families of
# run_dual_model.sbatch, default "qwen"; e.g. BASELINES="qwen llama").
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

export PDF_DIR="${PDF_DIR:-gold_pdfs}" FIGURES_DIR="${FIGURES_DIR:-gold_figures}" SET="${SET:-gold}"
export SEEDS="${SEEDS:-0 1 2}"
BASELINES="${BASELINES:-qwen}"
[ -n "${SBATCH_ACCOUNT:-}" ] || { echo "[!] SBATCH_ACCOUNT is not set (export SBATCH_ACCOUNT=<your_project>)"; exit 1; }
ls "$PDF_DIR"/*.pdf > /dev/null 2>&1 || { echo "[!] No PDFs in '$PDF_DIR'"; exit 1; }

DEP=()
if ls -d "$FIGURES_DIR"/*/ > /dev/null 2>&1; then
    echo "[~] '$FIGURES_DIR' already has figures: no crop job."
else
    CROP=$(sbatch --parsable --export=ALL scripts/qwen38/crop_figures.sbatch)
    echo "[+] crop_figures:  $CROP"
    DEP=(--dependency="afterok:$CROP")
fi

EVAL=$(sbatch --parsable "${DEP[@]}" --export=ALL scripts/qwen38/run_eval.sbatch)
echo "[+] qwen38 eval:   $EVAL"
ALL_IDS="${CROP:-} $EVAL"

for family in $BASELINES; do
    ckpt="full_analyze_checkpoints/${family}_${SET}"
    prev=""
    for seed in $SEEDS; do
        if [ -z "$prev" ]; then
            # First seed: digitizes the figures into $ckpt.
            id=$(FAMILY="$family" LOCAL_SEED="$seed" LATEX_DIR=none CHECKPOINT_DIR="$ckpt" \
                 OUTPUT="${family}_baseline_${SET}_s${seed}.jsonl" EXTRA_ARGS="--figures-dir $FIGURES_DIR" \
                 sbatch --parsable --time=06:00:00 "${DEP[@]}" --export=ALL scripts/run_dual_model.sbatch)
            first="$id"
        else
            # Other seeds: start from a copy of the first seed's figure analysis.
            id=$(FAMILY="$family" LOCAL_SEED="$seed" LATEX_DIR=none CHECKPOINT_DIR="${ckpt}_s${seed}" \
                 SEED_CHECKPOINT_DIR="$ckpt" OUTPUT="${family}_baseline_${SET}_s${seed}.jsonl" \
                 EXTRA_ARGS="--figures-dir $FIGURES_DIR" \
                 sbatch --parsable --time=06:00:00 --dependency="afterok:$first" --export=ALL scripts/run_dual_model.sbatch)
        fi
        prev="$id"
        ALL_IDS="$ALL_IDS $id"
        echo "[+] baseline $family seed $seed: $id"
    done
done

# Every job's logs: qwen38_* (crop/eval + one log per configuration),
# dual_model_* and vision_server_* (baselines), probe_* (vision probe).
REMOTE="${REMOTE:-$WORK/paper_analyzer}"
LOG_SOURCES=""
for id in $ALL_IDS; do LOG_SOURCES="$LOG_SOURCES :'$REMOTE/logs/*_${id}*'"; done
echo
echo "Monitor:   squeue -u \$USER"
echo "           tail -f $(for id in $ALL_IDS; do ls logs/*_"$id".out 2>/dev/null || echo "logs/<job>_$id.out"; done | tr '\n' ' ')"
echo "           (a job's .out appears only once it leaves the queue: qwen38_eval_<id>.out, dual_model_<id>.out)"
echo "When all jobs are done, from the LOCAL PC:"
echo "  mkdir -p logs_cineca/qwen38 && rsync -av <user>@login.leonardo.cineca.it:'$REMOTE/*_${SET}_*s[0-9]*.jsonl' $LOG_SOURCES logs_cineca/qwen38/"
