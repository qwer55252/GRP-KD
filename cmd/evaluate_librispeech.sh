#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grp_kd.evaluate --model "${MODEL:-outputs/ver5-ls100/student.nemo}" \
  --manifest "dev-clean=${MANIFEST_DIR:-data/manifests}/dev-clean.jsonl" \
  --manifest "dev-other=${MANIFEST_DIR:-data/manifests}/dev-other.jsonl" \
  --manifest "test-clean=${MANIFEST_DIR:-data/manifests}/test-clean.jsonl" \
  --manifest "test-other=${MANIFEST_DIR:-data/manifests}/test-other.jsonl" \
  --output "${RESULTS:-outputs/ver5-ls100/librispeech.json}" "$@"
