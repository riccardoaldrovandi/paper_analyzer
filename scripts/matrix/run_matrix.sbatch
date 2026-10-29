#!/bin/bash
#SBATCH --job-name=matrix
# No --account here: set SBATCH_ACCOUNT in your shell, as for run_dual_model.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/matrix_%j.out
#SBATCH --error=logs/matrix_%j.err

# The model comparison with the v2 prompt: ONE model per job, every source x
# mode x seed on the 10 gold papers, one vLLM server for all of them.
#   sbatch --export=ALL,MODEL=qwen38   scripts/matrix/run_matrix.sbatch
#   sbatch --export=ALL,MODEL=gemma4   scripts/matrix/run_matrix.sbatch
#   sbatch --export=ALL,MODEL=mistral4 scripts/matrix/run_matrix.sbatch
# Qwen3.8 uses 2 GPUs (TP=2): to queue for half a node, add
#   --gres=gpu:2 --cpus-per-task=16 --mem=240G
# Gemma 4 needs the 4 GPUs (TP=4) for a 128k context. Mistral too: with TP=2
# its Pixtral encoder ran out of memory on a single call with 12 page images
# ("Tried to allocate 7.12 GiB", job 62440420), so here it runs with TP=4 and
# GPU_MEM_UTIL 0.80 (headroom for the vision encoder's activations).
#
# SOURCES  pdf    gold_pdfs/ (figures for mode A: gold_figures/, the crops used so far)
#          latex  gold_latex/ (9 papers: anti-kt has no source; figures prepared
#                 locally by scripts/matrix/prepare_latex_figures.py)
# MODES    pipeline  A: the model digitizes each figure, PySR, final text call
#          single    B: ONE multimodal call, figure pages (PDF) or figure files
#                    (LaTeX) attached, no figure analysis of ours
#          libs      C: figure data read by libraries only (vector paths,
#                    pgfplots, tables), PySR, final text call
# CONFIGS  think (default): each model's reasoning mode with its model card
#          sampling and NO repetition/frequency penalties (with the old
#          defaults 1.15/0.3, Mistral's 3k-token v2 answer misspelled names to
#          avoid repeating them: "GQR", "MQAR", job 62435175)
#          nothink: reasoning off, model card sampling, no penalties
#          hybrid    B + C (prompt v2.1 only): the images of the single call AND
#                    the library-read figure data of libs, one multimodal call
# PROMPT_VERSION (v2): v2.1 = curves per figure, authors-only limitations and
#          datasets, PySR field filled by the code, figure crops + pages in the
#          single/hybrid call (src/prompt_v2.py). Use another OUT_DIR and CKPT_TAG.
# VERIFY_FIGURES=1 (v2.1, single/hybrid): second call checking the "figures"
#          entries against the images; the record keeps the list before it.
# Overrides: SOURCES, MODES, CONFIGS, SEEDS (0 1 2), MAX_PARALLEL (6), OUT_DIR (matrix),
# CKPT_TAG (checkpoint dirs full_analyze_checkpoints/matrix<tag>_...), MAX_IMAGES
# (12, images per single call), EXTRA_ARGS.
# Outputs: $OUT_DIR/<model>_<source>_<mode>_<config>_s<seed>.jsonl, provider
# "<model> [v2 <source> <mode> <config>]". Resumable: resubmit to continue.

module purge
module load gcc/12.2.0   # PySR/Julia needs GLIBCXX_3.4.26
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MODEL="${MODEL:?set MODEL=qwen38|gemma4|mistral4}"
SOURCES="${SOURCES:-pdf latex}"
MODES="${MODES:-pipeline single libs}"
CONFIGS="${CONFIGS:-think}"
SEEDS="${SEEDS:-0 1 2}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
OUT_DIR="${OUT_DIR:-matrix}"
CKPT_TAG="${CKPT_TAG:-}"
export PROMPT_VERSION="${PROMPT_VERSION:-v2}" VERIFY_FIGURES="${VERIFY_FIGURES:-0}"
case "$PROMPT_VERSION" in v2|v2.1) ;; *) echo "[!] PROMPT_VERSION must be v2 or v2.1"; exit 1 ;; esac
case " $MODES " in *" hybrid "*) [ "$PROMPT_VERSION" = v2.1 ] || { echo "[!] MODES hybrid needs PROMPT_VERSION=v2.1"; exit 1; } ;; esac
export MAX_IMAGES="${MAX_IMAGES:-12}" SINGLE_CALL_MAX_IMAGES="${MAX_IMAGES:-12}"
mkdir -p "$OUT_DIR"

for d in gold_pdfs gold_figures gold_latex gold_figures_latex gold_vector_latex; do
    [ -d "$d" ] || { echo "[!] '$d' missing: rsync it from the PC (see scripts/matrix/prepare_latex_figures.py)"; exit 1; }
done

source scripts/vllm_server.sh
case "$MODEL" in
    qwen38)   source scripts/qwen38/common.sh ;;
    gemma4)   source scripts/gemma4/common.sh; export GEMMA4_TP="${GEMMA4_TP:-4}" ;;
    mistral4) export MISTRAL4_TP="${MISTRAL4_TP:-4}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"; source scripts/mistral4/common.sh ;;
    *) echo "[!] Unknown MODEL '$MODEL'"; exit 1 ;;
esac
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$MODEL" in
    qwen38)   qwen38_start_and_wait 131072 || exit 1 ;;
    gemma4)   gemma4_start_and_wait 131072 || exit 1 ;;
    mistral4) mistral4_start_and_wait 131072 || exit 1 ;;
esac
source "$WORK/venv_vllm/bin/activate"
export KEEP_CHECKPOINTS=1 PYTHONUNBUFFERED=1
echo "[~] prompt $PROMPT_VERSION, verify figures $VERIFY_FIGURES, modes: $MODES, sources: $SOURCES, configs: $CONFIGS, seeds: $SEEDS -> $OUT_DIR/"

# model_env <config>: pipeline environment + sampling of one model and config.
# Common to all: 128k context, 32k output tokens (reasoning included), no penalties.
model_env() {
    export LOCAL_MAX_MODEL_LEN=131072 LOCAL_MAX_NEW_TOKENS=32768 \
           LOCAL_REPETITION_PENALTY=1.0 LOCAL_FREQUENCY_PENALTY=0 LOCAL_PRESENCE_PENALTY=0
    case "$MODEL:$1" in
        qwen38:think)     qwen38_pipeline_env; export LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=20 ;;
        qwen38:nothink)   qwen38_pipeline_env; export LOCAL_ENABLE_THINKING=0 LOCAL_TEMPERATURE=0.7 LOCAL_TOP_P=0.8 LOCAL_TOP_K=20 ;;
        gemma4:think)     gemma4_pipeline_env; export LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=64 \
                                                      LOCAL_EXTRA_BODY='{"skip_special_tokens": false}' ;;
        gemma4:nothink)   gemma4_pipeline_env; export LOCAL_ENABLE_THINKING=0 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=64 \
                                                      LOCAL_EXTRA_BODY='{"skip_special_tokens": false}' ;;
        # Model card: temperature 0.7 with reasoning, 0.0-0.7 without.
        mistral4:think)   export MISTRAL4_REASONING=high; mistral4_pipeline_env; export LOCAL_TEMPERATURE=0.7 ;;
        mistral4:nothink) export MISTRAL4_REASONING=none; mistral4_pipeline_env; export LOCAL_TEMPERATURE=0.3 ;;
        *) echo "[!] Unknown config '$1' for $MODEL"; return 1 ;;
    esac
}

if [ "$MODEL" = mistral4 ]; then
    # Broken UTF-8 (U+FFFD) in Mistral's answers of the first matrix: which server setting causes it.
    ( model_env think >/dev/null 2>&1; python scripts/mistral4/utf8_check.py ) 2>&1 | sed 's/^/[utf8] /'
fi

# source_args <source> <mode>: export-dataset inputs of one source and mode.
source_args() {
    case "$1:$2" in
        pdf:pipeline)   echo "--pdf-dir gold_pdfs --latex-dir none --figures-dir gold_figures" ;;
        # v2.1: the single/hybrid call also gets the figure crops (src/prompt_v2.py).
        pdf:single|pdf:hybrid) echo "--pdf-dir gold_pdfs --latex-dir none$([ "$PROMPT_VERSION" = v2.1 ] && echo " --figures-dir gold_figures")" ;;
        pdf:*)          echo "--pdf-dir gold_pdfs --latex-dir none" ;;
        latex:libs)     echo "--pdf-dir none --latex-dir gold_latex --vector-dir gold_vector_latex" ;;
        latex:hybrid)   echo "--pdf-dir none --latex-dir gold_latex --figures-dir gold_figures_latex --vector-dir gold_vector_latex" ;;
        latex:*)        echo "--pdf-dir none --latex-dir gold_latex --figures-dir gold_figures_latex" ;;
    esac
}

# run_one <source> <mode> <config> <seed> <checkpoint dir>: one export-dataset run, in a subshell.
run_one() {
    local name="${MODEL}_$1_$2_$3_s$4"
    (
        model_env "$3" || exit 1
        export LOCAL_RUN_TAG="$PROMPT_VERSION $1 $2 $3" LOCAL_SEED="$4" CHECKPOINT_DIR="$5"
        python main.py export-dataset --model local --mode "$2" $(source_args "$1" "$2") \
            --output "$OUT_DIR/$name.jsonl" $EXTRA_ARGS
    ) > "logs/matrix_${name}_${SLURM_JOB_ID}.log" 2>&1
    echo "[+] $(date +%H:%M) $name: $(wc -l 2>/dev/null < "$OUT_DIR/$name.jsonl" || echo 0) record(s)," \
         "$(wc -l 2>/dev/null < "$OUT_DIR/$name.rejected.jsonl" || echo 0) rejected"
}

pids=()
running() { local n=0 p; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }
launch() { while [ "$(running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done; echo "[~] $(date +%H:%M) start $*"; run_one "$@" & pids+=($!); }

read -r FIRST_CONFIG _ <<< "$CONFIGS"
read -r FIRST_SEED _ <<< "$SEEDS"
ckpt_of() { echo "full_analyze_checkpoints/matrix${CKPT_TAG}_${MODEL}_$1_$2"; }

echo; echo "################ 1. First seed of every source x mode (figure analysis) ################"
for source in $SOURCES; do
    for mode in $MODES; do
        ckpt="$(ckpt_of "$source" "$mode")"
        mkdir -p "$ckpt"
        # PDF + pipeline: the figure analysis of this model's earlier gold runs
        # (same crops, same digitization), when present: identical input, less time.
        # Not for v2.1: its figure blocks also carry the curves' read points.
        if [ "$source:$mode" = pdf:pipeline ] && [ "$PROMPT_VERSION" = v2 ] && ls "full_analyze_checkpoints/${MODEL}_gold"/*.checkpoint.json > /dev/null 2>&1; then
            cp -n "full_analyze_checkpoints/${MODEL}_gold"/*.checkpoint.json "$ckpt"/
            echo "[~] pdf pipeline: reusing the figure analysis of full_analyze_checkpoints/${MODEL}_gold"
        fi
        launch "$source" "$mode" "$FIRST_CONFIG" "$FIRST_SEED" "$ckpt"
    done
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
pids=()

echo; echo "################ 2. Other configs and seeds (max $MAX_PARALLEL in parallel) ################"
for source in $SOURCES; do
    for mode in $MODES; do
        for config in $CONFIGS; do
            for seed in $SEEDS; do
                [ "$config" = "$FIRST_CONFIG" ] && [ "$seed" = "$FIRST_SEED" ] && continue
                base="$(ckpt_of "$source" "$mode")"
                ckpt="${base}_${config}_s${seed}"
                mkdir -p "$ckpt"
                # Same figure analysis for every seed: only the final call varies.
                cp -n "$base"/*.checkpoint.json "$base"/*.libs.json "$ckpt"/ 2>/dev/null
                launch "$source" "$mode" "$config" "$seed" "$ckpt"
            done
        done
    done
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

echo; echo "################ Summary ################"
for f in "$OUT_DIR/${MODEL}"_*_s*.jsonl; do
    case "$f" in *.rejected.jsonl) continue ;; esac
    echo "  $f: $(wc -l < "$f") record(s), $(wc -l 2>/dev/null < "${f%.jsonl}.rejected.jsonl" || echo 0) rejected"
done
python main.py evaluate --input $(ls "$OUT_DIR/${MODEL}"_*_s*.jsonl | grep -v rejected) --rubrics gold/rubrics.jsonl --figure-checks gold/figure_checks.jsonl \
    --report "logs/matrix_${MODEL}_${SLURM_JOB_ID}.report.jsonl" --bootstrap 2000 2>&1 | tail -40
grep -iE "KV cache size|maximum concurrency|error" "$SERVER_LOG" | head -10
