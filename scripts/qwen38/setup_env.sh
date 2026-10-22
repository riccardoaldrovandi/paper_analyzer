#!/bin/bash
# One-time setup of the vLLM runtime for Qwen3.8. Run on a LOGIN node.
#   bash scripts/qwen38/setup_env.sh                 # auto: venv if glibc >= 2.31, else container
#   MODE=container bash scripts/qwen38/setup_env.sh  # force the Apptainer image
#
# venv:      $WORK/venv_vllm_qwen38, torch 2.10.0+cu128 + vLLM 0.19.1 from PyPI.
# container: $WORK/containers/vllm-openai_v0.19.1.sif from docker://vllm/vllm-openai.
# Either way the old $WORK/venv_vllm (vLLM 0.7.3) is not touched.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/qwen38/common.sh

# getconf, not `ldd --version | head -1`: under pipefail, head closing the pipe
# early can SIGPIPE ldd and silently kill the script right here.
GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
glibc_ok() { [ "$(printf '%s\n' 2.31 "$GLIBC" | sort -V | head -1)" = 2.31 ]; }
echo "[~] glibc $GLIBC | apptainer: ${APPTAINER_BIN:-not found} | vLLM $VLLM_VERSION"

MODE="${MODE:-auto}"
if [ "$MODE" = auto ]; then
    if glibc_ok; then MODE=venv
    elif [ -n "$APPTAINER_BIN" ]; then MODE=container
    else echo "[!] glibc $GLIBC < 2.31 and no apptainer/singularity: vLLM $VLLM_VERSION cannot be installed here."; exit 1
    fi
fi
echo "[~] Mode: $MODE"

if [ "$MODE" = venv ]; then
    glibc_ok || { echo "[!] glibc $GLIBC < 2.31: the vLLM $VLLM_VERSION wheel will not install. Use MODE=container."; exit 1; }
    set +u   # module and venv activate scripts read unset variables
    module purge
    module load gcc/12.2.0
    # Same Python 3.11 as the working venv_vllm, so no python module to guess.
    PYTHON_BIN="${PYTHON_BIN:-$WORK_DIR/venv_vllm/bin/python}"
    [ -d "$QWEN38_VENV" ] || "$PYTHON_BIN" -m venv "$QWEN38_VENV"
    source "$QWEN38_VENV/bin/activate"
    set -u
    pip install -U pip
    # torch first, from the cu128 index: CUDA 12.8 runs on the 12.2 driver
    # (CUDA 12.x minor-version compatibility), the PyPI default might not.
    pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128
    # transformers pinned near vLLM 0.19.1's release (Apr 2026): an unpinned
    # install takes the latest 5.x, the same trap that broke vLLM 0.7.3.
    pip install "vllm==$VLLM_VERSION" "transformers==${TRANSFORMERS_VERSION:-5.6.2}"
    pip freeze > "$WORK_DIR/venv_vllm_qwen38_freeze_$(date +%F).txt"
else
    [ -n "$APPTAINER_BIN" ] || { echo "[!] apptainer/singularity not found on this node."; exit 1; }
    mkdir -p "$(dirname "$QWEN38_SIF")" "$WORK_DIR/.cache/apptainer" "$WORK_DIR/.cache/apptainer_tmp"
    # Image layers are several GB: keep cache and build dir off $HOME and /tmp.
    export APPTAINER_CACHEDIR="$WORK_DIR/.cache/apptainer" SINGULARITY_CACHEDIR="$WORK_DIR/.cache/apptainer"
    export APPTAINER_TMPDIR="$WORK_DIR/.cache/apptainer_tmp" SINGULARITY_TMPDIR="$WORK_DIR/.cache/apptainer_tmp"
    [ -f "$QWEN38_SIF" ] || "$APPTAINER_BIN" pull "$QWEN38_SIF" "docker://vllm/vllm-openai:v$VLLM_VERSION"
fi

# Checks that need no GPU: versions, Qwen3.8's architecture known to vLLM,
# and (if already downloaded) the model config readable by transformers.
QWEN38_RUNTIME="$MODE"
qwen38_init_run
"${QWEN38_RUN[@]}" python3 - "$QWEN38_MODEL_DIR" <<'EOF'
import os, sys
import torch, transformers, vllm
from vllm.model_executor.models.registry import ModelRegistry
print(f"[~] torch {torch.__version__} (CUDA {torch.version.cuda}) | vllm {vllm.__version__} | transformers {transformers.__version__}")
if "Qwen3_5ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
    sys.exit("[!] This vLLM does not know Qwen3_5ForConditionalGeneration (Qwen3.8's architecture).")
print("[+] Qwen3_5ForConditionalGeneration supported by vLLM")
model_dir = sys.argv[1]
if os.path.exists(os.path.join(model_dir, "config.json")):
    cfg = transformers.AutoConfig.from_pretrained(model_dir)
    print(f"[+] Model config readable: model_type={cfg.model_type}")
else:
    print("[~] Model not downloaded yet: config check skipped (run download_model.sh).")
EOF
echo "[+] Setup done ($MODE)."
