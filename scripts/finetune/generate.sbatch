#!/bin/bash
#SBATCH --job-name=ft_gen
# No --account here: set SBATCH_ACCOUNT in your shell, as for run_matrix.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ft_gen_%j.out
#SBATCH --error=logs/ft_gen_%j.err

# Training labels of the fine-tuning set (decided 2026-10-28): teacher Qwen3.8
# (reasoning on, model card sampling, no penalties), prompt v2.1, physics
# corpus built by scripts/finetune/build_corpus.py (rsync corpus_ft/ first).
# Every paper is analyzed SEEDS times; scripts/finetune/select_answers.py then keeps
# the best answer per paper for the manual review.
#   sbatch scripts/finetune/generate.sbatch
#   sbatch --export=ALL,MODE=hybrid,SEEDS="0 1 2" scripts/finetune/generate.sbatch
# MODE     single (default) | hybrid | libs | pipeline  (see run_matrix.sbatch;
#          set it from the v2.1 matrix results)
# SOURCE   pdf (default) | latex
# SEEDS    "0 1"; SHARDS (6): the corpus is split in SHARDS parts run in
#          parallel (one export-dataset each, same server); VERIFY_FIGURES (1)
# Outputs: $OUT_DIR/qwen38_<source>_<mode>_s<seed>_part<k>.jsonl (resumable:
# resubmit to continue).

module purge
module load gcc/12.2.0   # PySR/Julia needs GLIBCXX_3.4.26
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MODE="${MODE:-single}"
SOURCE="${SOURCE:-pdf}"
SEEDS="${SEEDS:-0 1}"
SHARDS="${SHARDS:-6}"
CORPUS="${CORPUS:-corpus_ft}"
OUT_DIR="${OUT_DIR:-ft_data}"
export PROMPT_VERSION=v2.1 VERIFY_FIGURES="${VERIFY_FIGURES:-1}" MAX_IMAGES="${MAX_IMAGES:-12}" SINGLE_CALL_MAX_IMAGES="${MAX_IMAGES:-12}"
SRC_DIR="$CORPUS/$([ "$SOURCE" = pdf ] && echo pdfs || echo latex)"
[ -d "$SRC_DIR" ] || { echo "[!] '$SRC_DIR' missing: rsync $CORPUS/ from the PC"; exit 1; }
mkdir -p "$OUT_DIR"

# Shards: SHARDS directories of symlinks, the papers dealt round-robin (stable order).
SHARD_ROOT="$CORPUS/shards_${SOURCE}_${SHARDS}"
if [ ! -d "$SHARD_ROOT" ]; then
    i=0
    for f in $(ls "$SRC_DIR" | sort); do
        k=$((i % SHARDS)); mkdir -p "$SHARD_ROOT/$k"; ln -s "$(realpath "$SRC_DIR/$f")" "$SHARD_ROOT/$k/$f"; i=$((i + 1))
    done
fi
echo "[~] $(ls "$SRC_DIR" | wc -l) paper(s) in $SRC_DIR, $SHARDS shard(s), mode $MODE, seeds $SEEDS, verify figures $VERIFY_FIGURES"

source scripts/vllm_server.sh
source scripts/qwen38/common.sh
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
qwen38_start_and_wait 131072 || exit 1
source "$WORK/venv_vllm/bin/activate"
export KEEP_CHECKPOINTS=1 PYTHONUNBUFFERED=1

run_part() {  # <seed> <shard>
    local name="qwen38_${SOURCE}_${MODE}_s$1_part$2"
    (
        qwen38_pipeline_env
        export LOCAL_MAX_MODEL_LEN=131072 LOCAL_MAX_NEW_TOKENS=32768 LOCAL_REPETITION_PENALTY=1.0 LOCAL_FREQUENCY_PENALTY=0 \
               LOCAL_PRESENCE_PENALTY=0 LOCAL_ENABLE_THINKING=1 LOCAL_TEMPERATURE=1.0 LOCAL_TOP_P=0.95 LOCAL_TOP_K=20 \
               LOCAL_RUN_TAG="ft v2.1 $SOURCE $MODE" LOCAL_SEED="$1" CHECKPOINT_DIR="full_analyze_checkpoints/ft_${SOURCE}_${MODE}_s$1"
        if [ "$SOURCE" = pdf ]; then dirs="--pdf-dir $SHARD_ROOT/$2 --latex-dir none"; else dirs="--pdf-dir none --latex-dir $SHARD_ROOT/$2"; fi
        python main.py export-dataset --model local --mode "$MODE" $dirs --output "$OUT_DIR/$name.jsonl" $EXTRA_ARGS
    ) > "logs/ft_gen_${name}_${SLURM_JOB_ID}.log" 2>&1
    echo "[+] $(date +%H:%M) $name: $(wc -l 2>/dev/null < "$OUT_DIR/$name.jsonl" || echo 0) record(s), $(wc -l 2>/dev/null < "$OUT_DIR/$name.rejected.jsonl" || echo 0) rejected"
}

pids=()
for seed in $SEEDS; do
    for k in $(seq 0 $((SHARDS - 1))); do
        echo "[~] $(date +%H:%M) start seed $seed shard $k"
        run_part "$seed" "$k" & pids+=($!)
    done
done
wait "${pids[@]}"

echo; echo "################ Summary ################"
cat "$OUT_DIR"/qwen38_${SOURCE}_${MODE}_s*_part*.jsonl 2>/dev/null | wc -l | xargs echo "records:"
cat "$OUT_DIR"/qwen38_${SOURCE}_${MODE}_s*_part*.rejected.jsonl 2>/dev/null | wc -l | xargs echo "rejected:"
grep -iE "KV cache size|maximum concurrency|error" "$SERVER_LOG" | head -10
