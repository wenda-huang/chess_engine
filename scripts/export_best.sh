#!/usr/bin/env bash
# Export the deployed checkpoint to ONNX (fp32 + dynamic int8).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-models/best_2.pt}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

source .venv/bin/activate 2>/dev/null || true
python -m cli export-onnx --checkpoint "$CKPT" --int8

echo ""
echo "Done. Serve with:"
echo "  export CHESSAI_CHECKPOINT=$CKPT"
echo "  export CHESSAI_INFER=onnx-int8"
echo "  python -m cli serve"
