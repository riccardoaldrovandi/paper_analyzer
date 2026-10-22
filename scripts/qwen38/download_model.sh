#!/bin/bash
# Downloads Qwen/Qwen3.8-27B (BF16, ~52 GB) to $MODELS_DIR/qwen3.8-27b.
# Run on a LOGIN node: compute nodes have no internet access.
# Resumable: if it is interrupted, just run it again.
#   bash scripts/qwen38/download_model.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/qwen38/common.sh

REPO="${REPO:-Qwen/Qwen3.8-27B}"
HF_BIN="$(command -v hf || echo "$WORK_DIR/venv_vllm/bin/hf")"
[ -x "$HF_BIN" ] || { echo "[!] 'hf' CLI not found (expected in $WORK_DIR/venv_vllm/bin)"; exit 1; }

mkdir -p "$QWEN38_MODEL_DIR"
echo "[~] $REPO -> $QWEN38_MODEL_DIR (~52 GB needed)"
df -h "$WORK_DIR" | tail -1

"$HF_BIN" download "$REPO" --local-dir "$QWEN38_MODEL_DIR"

# Every shard named in the index must be on disk, otherwise vLLM fails late,
# on the compute node, after the job has already waited in the queue.
python3 - "$QWEN38_MODEL_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
shards = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
size = sum(os.path.getsize(os.path.join(d, s)) for s in shards if s not in missing)
print(f"[{'!' if missing else '+'}] {len(shards) - len(missing)}/{len(shards)} shards present, {size / 1e9:.1f} GB")
if missing:
    print("[!] Missing:", ", ".join(missing), "-- run the script again to resume.")
    sys.exit(1)
EOF
