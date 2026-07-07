import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from engine.config import Config
from engine.inference import create_inference_runner
from engine.player import EnginePlayer

fen = "r1bqkbnr/pppp1pp1/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 3 3"
b = chess.Board(fen)
cfg = Config()
ckpt = "models/supervised_big.pt"

backends = [
    ("torch", {"CHESSAI_INFER": "torch"}),
    ("onnx", {"CHESSAI_INFER": "onnx", "CHESSAI_ONNX": "models/supervised_big.onnx"}),
    (
        "onnx-int8",
        {"CHESSAI_INFER": "onnx-int8", "CHESSAI_ONNX": "models/supervised_big.int8.onnx"},
    ),
]

for name, env in backends:
    os.environ.update(env)
    runner = create_inference_runner(config=cfg, checkpoint=ckpt)
    player = EnginePlayer(runner=runner, config=cfg)
    t0 = time.time()
    out = player.evaluate_position(b, simulations=160)
    dt = time.time() - t0
    print(f"{name}: {dt:.2f}s  value={out['value']}")
