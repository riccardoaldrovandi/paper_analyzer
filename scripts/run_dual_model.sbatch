#!/bin/bash
#SBATCH --job-name=paper_analyzer_dual
#SBATCH --account=IscrC_MultLLM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# Half a node (32 cores / ~512 GB / 4 GPUs) to match the 2 GPUs: with the
# default memory the 28 GB Qwen text model load got the vision server
# OOM-killed (job 59727080).
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dual_model_%j.out
#SBATCH --error=logs/dual_model_%j.err

# 2-GPU split: the text model (in-process via src/llm/local_client.py) and the
# vision model (served via `vllm serve`) each get their own dedicated A100,
# fully isolated via CUDA_VISIBLE_DEVICES -- see project-vision-model-next-steps
# memory for why this architecture was chosen.
#
# Model pair is chosen with FAMILY (qwen | llama | mistral | velvet), e.g.:
#   sbatch --export=ALL,FAMILY=llama scripts/run_dual_model.sbatch
# Optional overrides: TEXT_MODEL, VISION_MODEL, LOCAL_TEMPERATURE,
# LOCAL_MAX_MODEL_LEN, PDF_DIR, LATEX_DIR, OUTPUT, EXTRA_ARGS (e.g. "--max-figures 5").
# SEED_CHECKPOINT_DIR: copy another run's figure checkpoints into CHECKPOINT_DIR
# first, so this run reuses that run's figure analysis (same vision input for
# every text model -> only the text model changes in the comparison).

module purge
module load gcc/12.2.0   # PySR/Julia needs GLIBCXX_3.4.26
source "$WORK/venv_vllm/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/vllm_server.sh

FAMILY="${FAMILY:-qwen}"
case "$FAMILY" in
    qwen)    TEXT_MODEL="${TEXT_MODEL:-qwen2.5-14b}";      VISION_MODEL="${VISION_MODEL:-qwen2.5-vl-7b}"
             export LOCAL_VISION_GUIDED_JSON="${LOCAL_VISION_GUIDED_JSON:-1}" ;;
    llama)   TEXT_MODEL="${TEXT_MODEL:-llama3.1-8b}";      VISION_MODEL="${VISION_MODEL:-llava-next-llama3-8b}"
             export LOCAL_VISION_GUIDED_JSON="${LOCAL_VISION_GUIDED_JSON:-1}" ;;
    mistral) TEXT_MODEL="${TEXT_MODEL:-mistral-nemo-12b}"; VISION_MODEL="${VISION_MODEL:-pixtral-12b}"
             export LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.3}"        # Mistral-Nemo's recommended value
             # Pixtral's Mistral-native tokenizer breaks guided decoding in vLLM 0.7.3
             # (run v7: every figure lost), so the vision model runs unconstrained.
             export LOCAL_VISION_GUIDED_JSON="${LOCAL_VISION_GUIDED_JSON:-0}" ;;
    # Almawave Velvet-14B is text only (no Velvet vision model exists): paired
    # with the best measured vision model, the same one as the qwen family.
    velvet)  TEXT_MODEL="${TEXT_MODEL:-velvet-14b}";       VISION_MODEL="${VISION_MODEL:-qwen2.5-vl-7b}"
             export LOCAL_VISION_GUIDED_JSON="${LOCAL_VISION_GUIDED_JSON:-1}" ;;
    *) echo "[!] Unknown FAMILY '$FAMILY' (qwen | llama | mistral | velvet)"; exit 1 ;;
esac

export LOCAL_MODEL_PATH="$MODELS_DIR/$TEXT_MODEL"
export LOCAL_VISION_MODEL_PATH="$MODELS_DIR/$VISION_MODEL"
# Port derived from the job ID: two jobs can land on the same node (4 GPUs,
# 2 per job), and a fixed port would let one job's readiness check find --
# and then send its figures to -- the OTHER job's vision server.
VISION_PORT="${VISION_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export LOCAL_VISION_SERVER_URL="http://localhost:$VISION_PORT/v1"
# One checkpoint dir per vision model: checkpoints hold that model's figure
# analysis, and must never be reused by another model in the comparison.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-full_analyze_checkpoints/$VISION_MODEL}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"
if [ -n "$SEED_CHECKPOINT_DIR" ]; then
    if ! ls "$SEED_CHECKPOINT_DIR"/*.checkpoint.json > /dev/null 2>&1; then
        echo "[!] SEED_CHECKPOINT_DIR '$SEED_CHECKPOINT_DIR' has no checkpoints"; exit 1
    fi
    mkdir -p "$CHECKPOINT_DIR"
    cp -n "$SEED_CHECKPOINT_DIR"/*.checkpoint.json "$CHECKPOINT_DIR"/
    echo "[~] Seeded $CHECKPOINT_DIR with $(ls "$SEED_CHECKPOINT_DIR"/*.checkpoint.json | wc -l) figure checkpoint(s) from $SEED_CHECKPOINT_DIR"
fi
OUTPUT="${OUTPUT:-training_data_${FAMILY}.jsonl}"

echo "[~] Family: $FAMILY | text: $TEXT_MODEL | vision: $VISION_MODEL | temperature: ${LOCAL_TEMPERATURE:-0.2} | vision guided JSON: $LOCAL_VISION_GUIDED_JSON"
echo "[~] Checkpoints: $CHECKPOINT_DIR | output: $OUTPUT | vision port: $VISION_PORT"
nvidia-smi -L

# Use the GPUs SLURM actually assigned to this job (on a shared node they may
# not be 0 and 1), one for the text model and one for the vision model.
IFS=, read -r TEXT_GPU VISION_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [ -z "$VISION_GPU" ]; then echo "[!] Need 2 GPUs, got CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"; exit 1; fi
echo "[~] GPUs assigned by SLURM: '$CUDA_VISIBLE_DEVICES' -> text: $TEXT_GPU, vision: $VISION_GPU"

# Always stop the vision server, even if the pipeline crashes or the job is cancelled.
trap stop_vision_server EXIT

# Vision GPU, isolated: OpenAI-compatible vLLM server for the vision model.
start_vision_server "$VISION_MODEL" "$VISION_GPU" "$VISION_PORT" || exit 1
# Wait before starting the pipeline -- otherwise the first
# digitize_plot_local() call would fail against a server still loading.
wait_for_vision_server "$VISION_PORT" || { echo "[!] Aborting before wasting GPU-hours on the main job."; exit 1; }

# Text GPU, isolated: main pipeline process, text model loaded in-process;
# figures are routed to the vision server above over HTTP.
export CUDA_VISIBLE_DEVICES="$TEXT_GPU"
echo "[~] Pipeline CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python main.py export-dataset --model local \
    --pdf-dir "${PDF_DIR:-downloaded_pdfs}" --latex-dir "${LATEX_DIR:-downloaded_latex}" \
    --output "$OUTPUT" $EXTRA_ARGS
