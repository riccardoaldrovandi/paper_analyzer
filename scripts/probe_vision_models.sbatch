#!/bin/bash
#SBATCH --job-name=vision_probe
#SBATCH --account=IscrC_MultLLM
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/vision_probe_%j.out
#SBATCH --error=logs/vision_probe_%j.err

# Runs scripts/vision_probe.py against each vision model in turn, on one GPU
# (one server at a time). Each model: ~2-5 min startup + ~1-2 min probing.
# Override the list with e.g.: sbatch --export=ALL,MODELS="idefics3-8b-llama3" ...

module purge
module load gcc/12.2.0
source "$WORK/venv_vllm/bin/activate"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source scripts/vllm_server.sh

MODELS="${MODELS:-idefics3-8b-llama3 llava-next-llama3-8b qwen2.5-vl-7b pixtral-12b}"
# Job-specific port, so this never talks to another job's server on a shared node.
VISION_PORT="${VISION_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
export LOCAL_VISION_SERVER_URL="http://localhost:$VISION_PORT/v1"
trap stop_vision_server EXIT
nvidia-smi -L

for model in $MODELS; do
    echo; echo "################ $model ################"
    start_vision_server "$model" "${CUDA_VISIBLE_DEVICES%%,*}" "$VISION_PORT" || continue
    if wait_for_vision_server "$VISION_PORT"; then
        LOCAL_VISION_MODEL_PATH="$MODELS_DIR/$model" python scripts/vision_probe.py --out "logs/probe_${model}_${SLURM_JOB_ID}.json"
    fi
    # Which chat template / content format vLLM actually used for this model.
    grep -i "chat template\|content format" "logs/vision_server_${model}_${SLURM_JOB_ID}.log" | head -5
    stop_vision_server
done
