#!/bin/bash
#SBATCH --job-name=gemma4_eval
# No --account here: set SBATCH_ACCOUNT in your shell, as for run_dual_model.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gemma4_eval_%j.out
#SBATCH --error=logs/gemma4_eval_%j.err

# Everything needed to evaluate Gemma 4 31B, in one job on 2 GPUs. ONE vLLM
# server (TP=2) reads both text and figures; the pipeline runs in venv_vllm
# and talks to it over HTTP. Submitted by submit_all.sh (after the figures
# exist), or:  sbatch scripts/gemma4/run_eval.sbatch
#
#  0. vision probe (scripts/vision_probe.py): same probe as the older vision
#     models, for the vision comparison table.
#  1. first CONFIG x first SEED: analyzes the figures (Gemma 4 digitization +
#     PySR) and keeps them as checkpoints.
#  2. every other CONFIG x SEED, up to MAX_PARALLEL at a time, each on a copy
#     of those checkpoints: same figure analysis for all, only the text
#     generation changes.
#
# CONFIGS (each a separate provider in `evaluate`, via LOCAL_RUN_TAG):
#   ctx32k   same settings as every earlier model (temperature 0.2, penalties,
#            32k context): the controlled comparison with Qwen2.5/Llama.
#   ctx128k  same, 128k context: about half the papers no longer truncated.
#   think    thinking ON with the model card's thinking sampling, 128k context.
# Overrides: PDF_DIR, FIGURES_DIR, SET (output name prefix), SEEDS, CONFIGS,
# MAX_PARALLEL, EXTRA_ARGS, EXPORT_SKIP_REJECTED=1 (on resume, do not retry
# papers already rejected). Outputs: gemma4_<SET>_<config>_s<seed>.jsonl.
# Resumable: resubmitting skips papers already written and figures already done.

module purge
module load gcc/12.2.0   # PySR/Julia needs GLIBCXX_3.4.26
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/gemma4/common.sh
source scripts/vllm_server.sh   # wait_for_vision_server / stop_vision_server
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PDF_DIR="${PDF_DIR:-gold_pdfs}"
FIGURES_DIR="${FIGURES_DIR:-gold_figures}"
SET="${SET:-gold}"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-ctx32k ctx128k think}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
if ! ls -d "$FIGURES_DIR"/*/ > /dev/null 2>&1; then
    echo "[!] No figures in '$FIGURES_DIR': run scripts/qwen38/crop_figures.sbatch first (submit_all.sh does)."
    exit 1
fi

# 131072 = the largest context any CONFIG uses; the client truncates each
# config to its own LOCAL_MAX_MODEL_LEN, so one server serves them all.
export GEMMA4_TP="${GEMMA4_TP:-4}"
gemma4_start_and_wait 131072 || exit 1

source "$WORK/venv_vllm/bin/activate"
gemma4_pipeline_env
export KEEP_CHECKPOINTS=1

echo; echo "################ 0. Vision probe ################"
python scripts/vision_probe.py --out "logs/probe_gemma4-31b_${SLURM_JOB_ID}.json" \
    > "logs/gemma4_probe_${SLURM_JOB_ID}.log" 2>&1 \
    && echo "[+] Probe done: logs/probe_gemma4-31b_${SLURM_JOB_ID}.json" \
    || echo "[!] Probe failed (see logs/gemma4_probe_${SLURM_JOB_ID}.log); continuing."

# run_config <config> <seed> <checkpoint_dir>: one export-dataset run, in a
# subshell so each configuration's variables never leak into the next.
run_config() {
    local config="$1" seed="$2" out="gemma4_${SET}_$1_s$2.jsonl"
    (
        export LOCAL_RUN_TAG="$config" LOCAL_SEED="$seed" CHECKPOINT_DIR="$3"
        case "$config" in
            ctx32k)  export LOCAL_MAX_MODEL_LEN=32768 ;;
            ctx128k) export LOCAL_MAX_MODEL_LEN=131072 ;;
            # Gemma 4 defaults (generation_config.json): temperature 1.0, top_p 0.95,
            # top_k 64, no penalties; 32768 output tokens as for Qwen3.8 (the
            # reasoning counts toward them). skip_special_tokens=false is what
            # lets the server's gemma4 reasoning parser split thought from answer.
            think)   export LOCAL_MAX_MODEL_LEN=131072 LOCAL_ENABLE_THINKING=1 LOCAL_MAX_NEW_TOKENS=32768 \
                            LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=64 \
                            LOCAL_REPETITION_PENALTY=1.0 LOCAL_FREQUENCY_PENALTY=0 LOCAL_PRESENCE_PENALTY=0 \
                            LOCAL_EXTRA_BODY='{"skip_special_tokens": false}' ;;
            *) echo "[!] Unknown config '$config'"; exit 1 ;;
        esac
        python main.py export-dataset --model local --pdf-dir "$PDF_DIR" --latex-dir none \
            --figures-dir "$FIGURES_DIR" --output "$out" $EXTRA_ARGS
    ) > "logs/gemma4_${SET}_${config}_s${seed}_${SLURM_JOB_ID}.log" 2>&1
    echo "[+] $(date +%H:%M) $out: $(wc -l < "$out" 2>/dev/null || echo 0) record(s)," \
         "$(wc -l < "${out%.jsonl}.rejected.jsonl" 2>/dev/null || echo 0) rejected"
}

read -r FIRST_CONFIG _ <<< "$CONFIGS"
read -r FIRST_SEED _ <<< "$SEEDS"
FIG_CKPT="full_analyze_checkpoints/gemma4_${SET}"

echo; echo "################ 1. Figures + $FIRST_CONFIG seed $FIRST_SEED ################"
run_config "$FIRST_CONFIG" "$FIRST_SEED" "$FIG_CKPT"

echo; echo "################ 2. Other configurations (max $MAX_PARALLEL in parallel) ################"
# The configurations' PIDs. `jobs`/`wait` alone would also count the vLLM server
# (a background job too): only 3 ran in parallel, and the final `wait` hung
# until the time limit (job 61447923 idled ~10 h after the last run).
pids=()
running() { local n=0 p; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }
for config in $CONFIGS; do
    for seed in $SEEDS; do
        [ "$config" = "$FIRST_CONFIG" ] && [ "$seed" = "$FIRST_SEED" ] && continue
        ckpt="${FIG_CKPT}_${config}_s${seed}"
        mkdir -p "$ckpt"
        cp -n "$FIG_CKPT"/*.checkpoint.json "$ckpt"/ 2>/dev/null
        while [ "$(running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
        echo "[~] $(date +%H:%M) starting $config seed $seed"
        run_config "$config" "$seed" "$ckpt" &
        pids+=($!)
    done
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

echo; echo "################ Summary ################"
for f in gemma4_${SET}_*_s*.jsonl; do
    case "$f" in *.rejected.jsonl) continue ;; esac
    echo "  $f: $(wc -l < "$f") record(s), $(wc -l < "${f%.jsonl}.rejected.jsonl" 2>/dev/null || echo 0) rejected"
done
grep -iE "KV cache size|maximum concurrency|error" "$SERVER_LOG" | head -10
