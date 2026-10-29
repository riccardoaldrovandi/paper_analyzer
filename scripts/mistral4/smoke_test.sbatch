#!/bin/bash
#SBATCH --job-name=mistral4_smoke
# No --account here: set SBATCH_ACCOUNT in your shell, as for run_dual_model.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mistral4_smoke_%j.out
#SBATCH --error=logs/mistral4_smoke_%j.err

# Smoke test of Mistral Small 4 on vLLM 0.19.1 (container), from the repo root:
#   sbatch scripts/mistral4/smoke_test.sbatch
# Default variant: AWQ 4-bit on TP=2 (see common.sh: the FP8 weights give
# garbage on A100). The job asks for 4 GPUs so MISTRAL4_TP=4 / MISTRAL4_VARIANT=fp8
# also fit; with TP=2 only the first 2 are used.
#
#  1. the capability checks (plain text first, then the four) of scripts/qwen38/smoke_test.py (text with
#     reasoning off, text with reasoning on + reasoning parser, image, guided
#     JSON), with Mistral's request settings (reasoning_effort, not a chat
#     template switch; model card: temperature 0.7 with reasoning, 0.0-0.7 without);
#  2. /tokenize, which fit_to_context uses with the Mistral tokenizer;
#  3. one real gold paper (SMOKE_PAPER, default GQA) through the whole pipeline
#     with the v2 prompt (PROMPT_VERSION=v2): figures from gold_figures/
#     digitized by Mistral, PySR, final answer with the v2 guided-JSON schema,
#     then `evaluate` with the rubric. Skip it with SMOKE_PIPELINE=0.
#
# If the job stops at "mistral_common < 1.11.0", install a newer one once, on
# the login node, and resubmit (common.sh then picks it up by itself):
#   source scripts/mistral4/common.sh
#   apptainer exec "$QWEN38_SIF" pip install --no-deps --target "$MISTRAL4_PYDEPS" "mistral_common[image]>=1.11,<2"
#
# Overrides: MAX_MODEL_LEN (131072), MISTRAL4_TP, GPU_MEM_UTIL, ENFORCE_EAGER=1,
# MISTRAL4_EXTRA_ARGS, SMOKE_PAPER, SMOKE_PIPELINE, PDF_DIR, FIGURES_DIR.

module purge
module load gcc/12.2.0   # PySR/Julia needs GLIBCXX_3.4.26
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/mistral4/common.sh
source scripts/vllm_server.sh
qwen38_init_run || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

[ -f "$MISTRAL4_MODEL_DIR/params.json" ] || { echo "[!] Model not found in $MISTRAL4_MODEL_DIR: run scripts/mistral4/download_model.sh"; exit 1; }
echo "[~] Runtime: $(qwen38_runtime) | model: $MISTRAL4_MODEL_DIR | GPUs: '$CUDA_VISIBLE_DEVICES'"
mistral4_start_and_wait "${MAX_MODEL_LEN:-131072}" || exit 1

echo; echo "################ 1. Capability checks ################"
"${QWEN38_RUN[@]}" python3 scripts/qwen38/smoke_test.py --port "$PORT" \
    --figure scripts/probe_figures/flux_tube.png \
    --out "logs/mistral4_smoke_${SLURM_JOB_ID}.json" \
    --nothink-json '{"temperature": 0.1, "reasoning_effort": "none", "skip_special_tokens": false}' \
    --think-json '{"temperature": 0.7, "reasoning_effort": "high", "skip_special_tokens": false}'
STATUS=$?

echo; echo "################ 2. /tokenize ################"
curl -s "http://localhost:$PORT/tokenize" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MISTRAL4_MODEL_DIR\", \"prompt\": \"Grouped-query attention interpolates between MHA and MQA.\", \"add_special_tokens\": false}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"[+] /tokenize OK: {d['count']} tokens\")" \
    || { echo "[!] /tokenize failed: fit_to_context would not work"; STATUS=1; }

if [ "${SMOKE_PIPELINE:-1}" = 1 ]; then
    echo; echo "################ 3. One gold paper, prompt v2 ################"
    PDF_DIR="${PDF_DIR:-gold_pdfs}"
    FIGURES_DIR="${FIGURES_DIR:-gold_figures}"
    SMOKE_PAPER="${SMOKE_PAPER:-5ae6fb6b_GQA}"
    paper=$(ls "$PDF_DIR"/"$SMOKE_PAPER"*.pdf 2>/dev/null | head -1)
    if [ -z "$paper" ]; then
        echo "[!] No PDF matching '$SMOKE_PAPER*' in $PDF_DIR"; STATUS=1
    else
        # A directory with just that paper, so export-dataset processes only it.
        one="logs/mistral4_smoke_pdf_${SLURM_JOB_ID}"
        mkdir -p "$one" && ln -sf "$PWD/$paper" "$one/"
        out="logs/mistral4_smoke_v2_${SLURM_JOB_ID}.jsonl"
        (
            source "$WORK/venv_vllm/bin/activate"
            mistral4_pipeline_env
            # v2 answers are 2-3x longer than v1: 4096 tokens could cut them.
            export PYTHONUNBUFFERED=1  # the log fills live, not only at the end
            export PROMPT_VERSION=v2 LOCAL_RUN_TAG=smoke_v2 LOCAL_MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}" \
                   LOCAL_MAX_NEW_TOKENS=12288 CHECKPOINT_DIR="full_analyze_checkpoints/mistral4_smoke_${SLURM_JOB_ID}"
            python main.py export-dataset --model local --pdf-dir "$one" --latex-dir none \
                --figures-dir "$FIGURES_DIR" --output "$out" \
                && python main.py evaluate --input "$out" --rubrics gold/rubrics.jsonl \
                       --report "logs/mistral4_smoke_v2_${SLURM_JOB_ID}.report.jsonl" --bootstrap 200
        ) 2>&1 | tee "logs/mistral4_smoke_v2_${SLURM_JOB_ID}.log" | grep -E "^\[(\+|!)\]|L1|L2|recall|Generation ended|Guided" | tail -40
        [ -s "$out" ] && echo "[+] v2 record written: $out" || { echo "[!] No v2 record: see logs/mistral4_smoke_v2_${SLURM_JOB_ID}.log"; STATUS=1; }
    fi
fi

echo; echo "[~] Server facts from $SERVER_LOG:"
grep -iE "weights|KV cache|maximum concurrency|attention backend|MLA|marlin|fp8|error" "$SERVER_LOG" | head -25
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
exit $STATUS
