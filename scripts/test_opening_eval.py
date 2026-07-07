import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
from engine.config import Config
from engine.model import load_checkpoint
from engine.player import EnginePlayer

fen = "r1bqkbnr/pppp1pp1/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 3 3"
b = chess.Board(fen)
cfg = Config()
m, _ = load_checkpoint("models/supervised_big.pt", device="cpu")
p = EnginePlayer(m, cfg)

print("FEN:", fen)
print("Side to move:", "white" if b.turn == chess.WHITE else "black")
for sims in [40, 80, 160]:
    r = p.evaluate_position(b, simulations=sims)
    top = r["suggestions"][0]["san"] if r["suggestions"] else None
    print(
        f"sims={sims}: value={r['value']} net_value={r['net_value']} "
        f"win_prob={r['win_prob']} top={top}"
    )

eng = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)
info = eng.analyse(b, chess.engine.Limit(depth=16))
score = info["score"].pov(chess.WHITE)
print("Stockfish depth16:", score)
