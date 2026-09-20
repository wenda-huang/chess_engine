import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, numpy as np, torch
from engine import gpuchess as G, gpuplay as P
from engine.encoding import IDX_TO_MOVE, move_to_index
from engine.model import load_checkpoint
dev = "cuda"
net, _ = load_checkpoint("models/small_10x128.pt", device=dev)
buf = P.GpuReplayBuffer(200_000, dev)
log = []
t = time.time()
stats = P.gpu_selfplay(net, buf, n_games=24, slots=8, sims=64, on_game=log.append)
print("selfplay", stats, f"{time.time()-t:.1f}s", "buffer", len(buf), "games logged", len(log))
print("lengths", [g["samples"] for g in log])
planes, policy, z = buf.sample_batch(2000)
print("z values", sorted(set(z.tolist())), "policy sums", float(policy.sum(1).min()), float(policy.sum(1).max()))
# each sampled position: policy support must be legal moves of the board rebuilt from planes
bad = 0
pl = planes.cpu().numpy(); po = policy.cpu().numpy()
for i in range(300):
    b = chess.Board(None)
    for k in range(12):
        for s in np.flatnonzero(pl[i, k].reshape(64)):
            b.set_piece_at(int(s), chess.Piece(k % 6 + 1, chess.WHITE if k < 6 else chess.BLACK))
    b.turn = bool(pl[i, 17, 0, 0])
    r = 0
    if pl[i, 12, 0, 0]: r |= chess.BB_H1
    if pl[i, 13, 0, 0]: r |= chess.BB_A1
    if pl[i, 14, 0, 0]: r |= chess.BB_H8
    if pl[i, 15, 0, 0]: r |= chess.BB_A8
    b.castling_rights = r
    e = np.flatnonzero(pl[i, 16].reshape(64))
    b.ep_square = int(e[0]) if len(e) else None
    legal = {move_to_index(m) for m in b.legal_moves}
    sup = set(np.flatnonzero(po[i] > 0).tolist())
    if not sup or not sup <= legal: bad += 1
print("samples with illegal/empty policy support:", bad, "/ 300")

# arena smoke
rng = __import__("random").Random(1)
from engine.books import build_mixed_opening_pool
ops = build_mixed_opening_pool(rng, size=16, book=None, min_plies=6, max_plies=24, random_min_plies=4, random_max_plies=12)
t = time.time()
print("arena (same net vs itself):", P.gpu_arena(net, net, ops, sims=64), f"{time.time()-t:.1f}s")
