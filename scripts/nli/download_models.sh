#!/bin/bash
# Downloads every judge model into $NLI_HF_CACHE. Run on a LOGIN node (compute
# nodes have no internet). Resumable: run it again if it is interrupted.
#   bash scripts/nli/download_models.sh                 # small models (~5 GB)
#   WITH_BESPOKE=1 bash scripts/nli/download_models.sh  # + Bespoke-MiniCheck-7B (~8-15 GB)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/nli/common.sh

HF_BIN="$NLI_VENV/bin/hf"
[ -x "$HF_BIN" ] || HF_BIN="$(command -v hf || echo "$WORK_DIR/venv_vllm/bin/hf")"
[ -x "$HF_BIN" ] || { echo "[!] 'hf' CLI not found: run scripts/nli/setup_env.sh first"; exit 1; }
mkdir -p "$NLI_HF_CACHE"
df -h "$WORK_DIR" | tail -1

for repo in "${NLI_MODEL_REPOS[@]}"; do
    echo "[~] $repo"
    "$HF_BIN" download "$repo" --cache-dir "$NLI_HF_CACHE" > /dev/null
done
# Only the tokenizer/config files of the base models, not their weights.
for repo in "${NLI_TOKENIZER_REPOS[@]}"; do
    echo "[~] tokenizer of $repo"
    "$HF_BIN" download "$repo" --cache-dir "$NLI_HF_CACHE" \
        --include "*.json" "*.model" "*.txt" "spiece.model" "merges.txt" "vocab*" > /dev/null
done
if [ "${WITH_BESPOKE:-0}" = 1 ]; then
    echo "[~] $BESPOKE_REPO (~8-15 GB)"
    "$HF_BIN" download "$BESPOKE_REPO" --cache-dir "$NLI_HF_CACHE" > /dev/null
fi
echo "[+] Downloaded. Cache content:"
ls "$NLI_HF_CACHE" | grep '^models--' || true
du -sh "$NLI_HF_CACHE"
