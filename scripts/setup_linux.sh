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
install_torch() {
  pip install --upgrade pip
  if command -v nvidia-smi >/dev/null 2>&1; then
    # RTX 50-series (Blackwell sm_120) needs PyTorch >=2.7 built with CUDA 12.8 (cu128).
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    if echo "$GPU_NAME" | grep -qiE 'RTX 50|Blackwell'; then
      echo "==> Blackwell GPU detected ($GPU_NAME) — installing PyTorch cu128..."
      # Chess only needs torch (not torchvision). Stable cu128 supports sm_120.
      pip install torch --index-url https://download.pytorch.org/whl/cu128 \
        || pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
    else
      echo "==> Installing PyTorch (CUDA 12.4 wheels)..."
      pip install torch --index-url https://download.pytorch.org/whl/cu124
    fi
  else
    pip install torch
  fi
}

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  if python -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
cap = torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
if cap >= (12, 0) and archs and 'sm_120' not in archs and 'compute_120' not in ''.join(archs):
    raise SystemExit(1)
" 2>/dev/null; then
    echo "==> torch already installed with compatible CUDA"
  else
    echo "==> Reinstalling PyTorch (GPU arch mismatch)..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    install_torch
  fi
else
  echo "==> Installing PyTorch..."
  install_torch
fi

pip install -r requirements.txt

# --- 4. Sanity checks ---
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('capability', torch.cuda.get_device_capability())
    if hasattr(torch.cuda, 'get_arch_list'):
        print('arch_list', torch.cuda.get_arch_list())
    x = torch.randn(4, device='cuda')
    y = x @ x
    print('matmul ok', y.shape)
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
# Self-play uses many processes — pin BLAS/torch to 1 thread each.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
