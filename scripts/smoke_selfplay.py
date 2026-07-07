#!/usr/bin/env python3
"""Quick self-play smoke test — runs 1 game with 1 worker, logs everything."""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force torch inference (never ONNX in workers).
os.environ["CHESSAI_INFER"] = "torch"


def main() -> None:
    from engine.config import Config
    from engine.model import load_checkpoint
    from engine.player import EnginePlayer
    from train.selfplay import play_selfplay_game

    config = Config()
    ckpt = "models/supervised_big.pt"
    if not os.path.exists(ckpt):
        print(f"ERROR: missing {ckpt}")
        sys.exit(1)

    print("device:", config.device)
    print("stockfish:", config.stockfish_path)
    print("data:", config.data_dir)
    print("infer:", os.environ.get("CHESSAI_INFER", "default"))

    try:
        model, meta = load_checkpoint(ckpt, device="cpu")
        print("loaded checkpoint, meta:", meta)
        player = EnginePlayer(model, config)
        samples, result = play_selfplay_game(player, sims=40, temperature_moves=8)
        print(f"OK: {len(samples)} positions, white_result={result}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
