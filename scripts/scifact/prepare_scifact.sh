#!/bin/bash
# Downloads the SciFact data (allenai/scifact release: corpus.jsonl, claims_dev.jsonl)
# to $WORK/scifact. Run on a LOGIN node (compute nodes have no internet).
#   bash scripts/scifact/prepare_scifact.sh
set -euo pipefail
WORK_DIR="${WORK:?WORK is not set (it is on Leonardo)}"
DEST="$WORK_DIR/scifact"
mkdir -p "$DEST"
cd "$DEST"
URL="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
[ -f data.tar.gz ] || curl -fL -o data.tar.gz "$URL"
tar -xzf data.tar.gz
ls -la "$DEST" "$DEST"/data 2>/dev/null | head -20
test -f "$DEST/data/corpus.jsonl" && test -f "$DEST/data/claims_dev.jsonl" && echo "[+] SciFact ready in $DEST/data"
