#!/bin/bash
# Downloads google/gemma-4-31B-it (BF16, ~62 GB) to $MODELS_DIR/gemma4-31b.
# Run on a LOGIN node (compute nodes have no internet). Resumable.
# The model is gated: accept the license on its Hugging Face page and be logged
# in (`hf auth login`) before running.
#   bash scripts/gemma4/download_model.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/gemma4/common.sh

REPO="${REPO:-google/gemma-4-31B-it}"
HF_BIN="$(command -v hf || echo "$WORK_DIR/venv_vllm/bin/hf")"
[ -x "$HF_BIN" ] || { echo "[!] 'hf' CLI not found (expected in $WORK_DIR/venv_vllm/bin)"; exit 1; }

mkdir -p "$GEMMA4_MODEL_DIR"
echo "[~] $REPO -> $GEMMA4_MODEL_DIR (~62 GB needed)"
df -h "$WORK_DIR" | tail -1
"$HF_BIN" download "$REPO" --local-dir "$GEMMA4_MODEL_DIR"

python3 - "$GEMMA4_MODEL_DIR" <<'EOF'
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

# Facts the next steps depend on (thinking switch, sampling defaults, architecture).
echo; echo "---- config.json: architectures / model_type ----"
python3 -c "import json,sys;c=json.load(open('$GEMMA4_MODEL_DIR/config.json'));print(c.get('architectures'),c.get('model_type'),c.get('max_position_embeddings',c.get('text_config',{}).get('max_position_embeddings')))"
echo "---- generation_config.json ----"
cat "$GEMMA4_MODEL_DIR/generation_config.json" 2>/dev/null || echo "(none)"
echo "---- chat template: lines mentioning thinking ----"
{ cat "$GEMMA4_MODEL_DIR/chat_template.jinja" 2>/dev/null || python3 -c "import json;print(json.load(open('$GEMMA4_MODEL_DIR/tokenizer_config.json')).get('chat_template',''))"; } | grep -n -i -E "think|thought|channel" | head -20
