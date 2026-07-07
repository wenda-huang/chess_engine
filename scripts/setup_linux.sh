#!/usr/bin/env bash
# One-time setup on a fresh Linux machine (e.g. cloud GPU at /workspace/chess).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Project root: $ROOT"

# --- 1. System deps ---
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v stockfish >/dev/null 2>&1; then
    echo "==> Installing stockfish (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq stockfish
  fi
fi

STOCKFISH="${STOCKFISH_PATH:-$(command -v stockfish || true)}"
if [[ -z "$STOCKFISH" || ! -x "$STOCKFISH" ]]; then
  echo "ERROR: Stockfish not found. Install it or set STOCKFISH_PATH."
  exit 1
fi
echo "==> Stockfish: $STOCKFISH"

# --- 2. Python venv ---
if [[ ! -d .venv ]]; then
  echo "==> Creating .venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Python: $(python --version)"

# --- 3. PyTorch (CUDA if available) ---
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "==> torch already installed with CUDA"
else
  echo "==> Installing PyTorch..."
  if command -v nvidia-smi >/dev/null 2>&1; then
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu124
  else
    pip install --upgrade pip
    pip install torch
  fi
fi

pip install -r requirements.txt

# --- 4. Sanity checks ---
python -c "
import torch, chess, onnxruntime as ort
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('chess', chess.__version__)
print('onnxruntime', ort.__version__)
"

# --- 5. Verify transferred assets ---
check_optional() {
  local path="$1" label="$2"
  if [[ -e "$path" ]]; then
    echo "==> OK: $label ($path)"
  else
    echo "==> MISSING (optional): $label ($path)"
  fi
}

check_optional "models/supervised_big.pt" "trained model"
check_optional "data_d16" "4M depth-16 training data"
check_optional "books/opening.bin" "opening book"
check_optional "books/syzygy" "syzygy tablebases"
check_optional "models/supervised_big.int8.onnx" "ONNX int8 export"

# --- 6. Export ONNX if checkpoint exists but ONNX does not ---
if [[ -f models/supervised_big.pt && ! -f models/supervised_big.int8.onnx ]]; then
  echo "==> Exporting ONNX (one-time)..."
  python -m cli export-onnx --checkpoint models/supervised_big.pt --int8
fi

# --- 7. Write env file for convenience ---
ENV_FILE="$ROOT/.env.sh"
cat > "$ENV_FILE" <<EOF
# Source before running:  source .env.sh
export STOCKFISH_PATH="$STOCKFISH"
export CHESSAI_DATA="${CHESSAI_DATA:-$ROOT/data_d16}"
export CHESSAI_DEVICE="${CHESSAI_DEVICE:-cuda}"
export CHESSAI_INFER="${CHESSAI_INFER:-onnx-int8}"
export CHESSAI_ONNX="${CHESSAI_ONNX:-$ROOT/models/supervised_big.int8.onnx}"
export OPENING_BOOK="${OPENING_BOOK:-$ROOT/books/opening.bin}"
export SYZYGY_PATH="${SYZYGY_PATH:-$ROOT/books/syzygy}"
EOF
echo "==> Wrote $ENV_FILE"

# --- 8. Quick smoke test ---
if [[ -f models/supervised_big.pt ]]; then
  echo "==> Smoke test: load checkpoint + 80-sim eval..."
  source "$ENV_FILE"
  python -c "
import chess
from engine.config import Config
from engine.player import EnginePlayer
cfg = Config()
p = EnginePlayer(config=cfg, checkpoint='models/supervised_big.pt')
m, _ = p.select_move(chess.Board(), simulations=80)
print('best move:', m.uci())
"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  cd $ROOT"
echo "  source .venv/bin/activate"
echo "  source .env.sh"
echo "  python -m cli evaluate --checkpoint models/supervised_big.pt --games 4 --skill 5 --sims 120 --workers 4"
echo "  python -m cli selfplay --init models/supervised_big.pt --workers 14 --selfplay-device cpu ..."
