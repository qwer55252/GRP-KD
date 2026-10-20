#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grp_kd.train --config configs/ls100.json \
  --train-manifest "${MANIFEST_DIR:-data/manifests}/train-clean-100.jsonl" \
  --val-manifest "${MANIFEST_DIR:-data/manifests}/dev-clean.jsonl" \
  --output-dir "${OUTPUT_DIR:-outputs/ver5-ls100}" "$@"
