#!/bin/bash
# One-time setup of the groundedness-judge environment. Run on a LOGIN node.
#   bash scripts/nli/setup_env.sh                 # with vLLM (needed by Bespoke-MiniCheck-7B)
#   INSTALL_VLLM=0 bash scripts/nli/setup_env.sh  # without vLLM
# Creates $WORK/venv_nli; the other venvs are not touched.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/nli/common.sh

set +u   # module and venv activate scripts read unset variables
module purge
module load gcc/12.2.0
PYTHON_BIN="${PYTHON_BIN:-$WORK_DIR/venv_vllm/bin/python}"
[ -d "$NLI_VENV" ] || "$PYTHON_BIN" -m venv "$NLI_VENV"
source "$NLI_VENV/bin/activate"
set -u
pip install -U pip

# torch first, from the cu121 index (same build as venv_vllm).
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if [ "${INSTALL_VLLM:-1}" = 1 ]; then
    pip install vllm==0.7.3
fi
# Re-pin after vLLM: an unpinned transformers installs 5.x and numpy 2 breaks vLLM 0.7.3.
pip install "transformers==4.49.0" "numpy==1.26.4" sentencepiece protobuf accelerate \
    nltk datasets pandas openai tqdm python-dotenv "huggingface_hub[cli]"
# --no-deps: the package's own requirements are unpinned and would replace torch.
pip install --no-deps "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"

mkdir -p "$NLTK_DATA"
python -c "
import nltk
for pkg in ('punkt', 'punkt_tab'):
    nltk.download(pkg, download_dir='$NLTK_DATA')
"
pip freeze > "$WORK_DIR/venv_nli_freeze_$(date +%F).txt"
python -c "
import torch, transformers, minicheck
print(f'[+] torch {torch.__version__} | transformers {transformers.__version__} | minicheck imported')
"
echo "[+] Setup done: $NLI_VENV"
