#!/bin/bash
# Downloads Mistral Small 4 in the variant chosen by MISTRAL4_VARIANT (see
# common.sh): awq (default) = cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit
# (~71 GB) to $MODELS_DIR/mistral-small-4-119b-awq; fp8 = the official
# mistralai/Mistral-Small-4-119B-2603 (~121 GB) to $MODELS_DIR/mistral-small-4-119b.
# Only the Mistral-native format (consolidated-*.safetensors, params.json,
# tekken.json): both repos also hold the same weights in HF format
# (model-*.safetensors) that vLLM 0.19.1 cannot load anyway. Not gated.
# Run on a LOGIN node (compute nodes have no internet), inside tmux: it takes a
# while and must survive a dropped ssh session. Resumable: just run it again.
#   bash scripts/mistral4/download_model.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/mistral4/common.sh

REPO="${REPO:-$MISTRAL4_REPO}"
HF_BIN="$(command -v hf || echo "$WORK_DIR/venv_vllm/bin/hf")"
[ -x "$HF_BIN" ] || { echo "[!] 'hf' CLI not found (expected in $WORK_DIR/venv_vllm/bin)"; exit 1; }

mkdir -p "$MISTRAL4_MODEL_DIR"
echo "[~] $REPO (Mistral format only) -> $MISTRAL4_MODEL_DIR (~$([ "$MISTRAL4_VARIANT" = awq ] && echo 71 || echo 121) GB needed)"
df -h "$WORK_DIR" | tail -1
"$HF_BIN" download "$REPO" --local-dir "$MISTRAL4_MODEL_DIR" \
    --include "consolidated*" "params.json" "tekken.json" "SYSTEM_PROMPT.txt" "README.md" "generation_config.json"

# Every shard named in the index must be on disk, otherwise vLLM fails late,
# on the compute node, after the job has already waited in the queue.
python3 - "$MISTRAL4_MODEL_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
shards = sorted(set(json.load(open(os.path.join(d, "consolidated.safetensors.index.json")))["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
size = sum(os.path.getsize(os.path.join(d, s)) for s in shards if s not in missing)
print(f"[{'!' if missing else '+'}] {len(shards) - len(missing)}/{len(shards)} shards present, {size / 1e9:.1f} GB")
for f in ("params.json", "tekken.json"):
    if not os.path.exists(os.path.join(d, f)):
        missing.append(f)
        print(f"[!] {f} missing")
if missing:
    print("[!] Run the script again to resume.")
    sys.exit(1)
p = json.load(open(os.path.join(d, "params.json")))
print(f"[+] params.json: {p['n_layers']} layers, {p['moe']['num_experts']} experts "
      f"({p['moe']['num_experts_per_tok']} active, {p['moe']['num_shared_experts']} shared), "
      f"vision encoder: {'yes' if p.get('vision_encoder') else 'NO'}, "
      f"quantization: {p.get('quantization') or (p.get('quantization_config') or {}).get('quant_method')}")
EOF
