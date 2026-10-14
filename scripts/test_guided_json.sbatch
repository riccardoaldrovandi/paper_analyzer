#!/bin/bash
#SBATCH --job-name=guided_json_test
# No --account here: set it once in your shell (sbatch reads it from the
# environment), e.g. in ~/.bashrc:  export SBATCH_ACCOUNT=<your_project>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/guided_test_%j.out
#SBATCH --error=logs/guided_test_%j.err

# Free vs schema-guided JSON (scripts/test_guided_json.py), on one GPU and one
# model at a time: first the text models, then the vision server.
# Overrides: INPUTS (records whose user messages are reused), TEXT_MODELS, VISION_MODELS.
#   sbatch --export=ALL,INPUTS=training_data_qwen_v5.jsonl scripts/test_guided_json.sbatch
#   sbatch --export=ALL,TEXT_MODELS=,VISION_MODELS="pixtral-12b qwen2.5-vl-7b" scripts/test_guided_json.sbatch   # vision only

module purge
module load gcc/12.2.0
source "$WORK/venv_vllm/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/vllm_server.sh

# The same inputs for every text model, so only guided vs free changes.
INPUTS="${INPUTS:-training_data_qwen_v5.jsonl}"
TEXT_MODELS="${TEXT_MODELS-qwen2.5-14b mistral-nemo-12b}"   # set to "" to skip the text tests
VISION_MODELS="${VISION_MODELS-pixtral-12b}"
GPU="${CUDA_VISIBLE_DEVICES%%,*}"
VISION_PORT="${VISION_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export LOCAL_VISION_SERVER_URL="http://localhost:$VISION_PORT/v1"
trap stop_vision_server EXIT
nvidia-smi -L

for model in $TEXT_MODELS; do
    echo; echo "################ TEXT: $model ################"
    temperature=0.2
    [ "$model" = "mistral-nemo-12b" ] && temperature=0.3   # Mistral-Nemo's recommended value, as in run_dual_model.sbatch
    CUDA_VISIBLE_DEVICES="$GPU" LOCAL_MODEL_PATH="$MODELS_DIR/$model" LOCAL_TEMPERATURE="$temperature" \
        python scripts/test_guided_json.py text --inputs "$INPUTS" --out "logs/guided_test_${model}_${SLURM_JOB_ID}"
    # Which guided-decoding backend vLLM actually used (xgrammar or a fallback).
    grep -ih "xgrammar\|outlines\|guided" "logs/guided_test_${SLURM_JOB_ID}.err" | sort | uniq -c | head -5
done

for model in $VISION_MODELS; do
    echo; echo "################ VISION: $model ################"
    start_vision_server "$model" "$GPU" "$VISION_PORT" || continue
    if wait_for_vision_server "$VISION_PORT"; then
        LOCAL_VISION_MODEL_PATH="$MODELS_DIR/$model" \
            python scripts/test_guided_json.py vision --figures-dir comparison_figures --out "logs/guided_test_vision_${model}_${SLURM_JOB_ID}"
    fi
    grep -ih "xgrammar\|outlines\|guided" "logs/vision_server_${model}_${SLURM_JOB_ID}.log" | sort | uniq -c | head -5
    stop_vision_server
done
