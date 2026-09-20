"""Sanity/benchmark for engine.gpumcts: tactics, agreement with the python MCTS, speed."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess, numpy as np, torch
from engine import gpuchess as G, gpumcts as M
from engine.config import Config
from engine.model import load_checkpoint
from engine.player import EnginePlayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/small_10x128.pt"
net, _ = load_checkpoint(ckpt, device=dev); net.eval()
cfg = Config(); cfg.device = dev
py = EnginePlayer(net, cfg)

FENS = {
    "mate in 1": ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    "free queen": ("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", None),
    "hanging rook": ("4k3/8/8/3r4/8/8/3R4/4K3 w - - 0 1", "d2d5"),
}
sims = 400
boards = [chess.Board(f) for f, _ in FENS.values()] * 8
st = G.state_from_boards(boards, dev)
res = M.search(net, st, sims)
for i, (name, (fen, want)) in enumerate(FENS.items()):
    v = res.visits[i].cpu().numpy(); c = res.code[i].cpu().numpy()
    top = int(v.argmax()); mv = G.code_to_move(int(c[top]))
    pm, root = py.select_move(chess.Board(fen), simulations=sims, temperature=0)
    b = chess.Board(fen)
    print(f"{name}: gpu={b.san(mv)} ({v[top]/v.sum():.2f} of visits, q={float(res.root_q[i]):.2f})  python={b.san(pm)} (q={root.Q:.2f})  expected={want}")

# agreement on opening/middlegame positions: does the GPU search pick the python search's move?
import random
rng = random.Random(3)
pos = []
for _ in range(24):
    b = chess.Board()
    for _ in range(rng.randint(6, 30)):
        if b.is_game_over(): break
        b.push(rng.choice(list(b.legal_moves)))
    if not b.is_game_over(): pos.append(b)
st = G.state_from_boards(pos, dev)
res = M.search(net, st, sims)
agree = 0; overlap = []
for i, b in enumerate(pos):
    v = res.visits[i].cpu().numpy(); c = res.code[i].cpu().numpy()
    g_move = G.code_to_move(int(c[int(v.argmax())]))
    pm, root = py.select_move(b, simulations=sims, temperature=0)
    agree += g_move == pm
    pv = {m: ch.N for m, ch in root.children.items()}
    gv = {G.code_to_move(int(c[j])): v[j] for j in range(len(v)) if v[j] > 0}
    tot_p, tot_g = sum(pv.values()), sum(gv.values())
    overlap.append(sum(min(pv.get(m, 0) / tot_p, gv.get(m, 0) / tot_g) for m in pv))
print(f"top-move agreement with python MCTS: {agree}/{len(pos)}; mean visit-distribution overlap {np.mean(overlap):.2f}")

# speed
for B in (64, 128, 256):
    st = G.start_state(B, dev)
    M.search(net, st, 50)  # warm
    torch.cuda.synchronize(); t = time.time()
    M.search(net, st, sims, add_noise=True)
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"B={B}: {dt:.2f}s per {sims}-sim move for all games -> {B/dt:.1f} moves/s")
