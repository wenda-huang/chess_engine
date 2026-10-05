#!/usr/bin/env python3
"""Candidate network vs Stockfish, with our side's search fully GPU-resident.

Our engine always searches via ``engine.gpumcts`` (batched GPU MCTS, the same path
used by self-play and the GPU arena) -- never the CPU Python MCTS in ``engine.mcts``.
Stockfish instances run as ordinary UCI subprocesses, which is how Stockfish always
runs; it has no GPU mode. Many games play concurrently so the GPU search stays
batched while a small pool of Stockfish processes handles the opponent moves.

Plays a short sweep across a few Stockfish "Skill Level" anchors (see SKILL_ELO)
and derives an absolute Elo estimate from the score at each anchor.

    python scripts/gpu_vs_stockfish.py models/best_small.pt --skills 4,7,10 --pairs 40
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
import torch

from engine import gpuchess as G
from engine import gpumcts as M
from engine.gpuplay import _game_over
from engine.model import load_checkpoint

# Rough anchor: Stockfish "Skill Level" UCI option -> approximate Elo (ballpark,
# community testing at short time controls -- same table train/evaluate.py uses).
SKILL_ELO = {
    0: 1350, 1: 1425, 2: 1500, 3: 1575, 4: 1650, 5: 1750, 6: 1850,
    7: 1950, 8: 2050, 9: 2150, 10: 2250, 12: 2450, 15: 2700, 20: 3000,
}


def _elo_diff(score: float, n: int) -> float:
    eps = 1.0 / (2 * n)
    score = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


def _random_opening(rng: random.Random, plies: int = 2) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        if b.is_game_over():
            break
        b.push(rng.choice(list(b.legal_moves)))
    return b


@torch.no_grad()
def run_match(
    net,
    stockfish_path: str,
    pairs: int,
    skill_level: int,
    movetime: float,
    sims: int,
    sf_workers: int,
    seed: int,
    c_puct: float = 1.5,
    amp: bool = True,
    max_plies: int = 240,
) -> dict:
    dev = next(net.parameters()).device
    net.eval()
    rng = random.Random(seed)
    openings = [_random_opening(rng) for _ in range(pairs)]
    S = 2 * pairs
    boards = [b for b in openings for _ in range(2)]   # slots 2i, 2i+1 share an opening
    st = G.state_from_boards(boards, dev)
    cand_white = (torch.arange(S, device=dev) % 2 == 0)
    ar = torch.arange(S, device=dev)
    ply = torch.zeros(S, dtype=torch.long, device=dev)
    hist = torch.zeros(S, max_plies + 2, dtype=torch.long, device=dev)
    hist[:, 0] = G.hash_state(st)
    done = torch.zeros(S, dtype=torch.bool, device=dev)
    result = torch.zeros(S, device=dev)

    engines = [chess.engine.SimpleEngine.popen_uci(stockfish_path) for _ in range(sf_workers)]
    for e in engines:
        try:
            e.configure({"Skill Level": skill_level, "Threads": 1})
        except Exception:
            pass
    pool = ThreadPoolExecutor(max_workers=sf_workers)

    def sf_move(args):
        eng, board = args
        try:
            r = eng.play(board, chess.engine.Limit(time=movetime))
            return r.move
        except Exception:
            return None

    t0 = time.time()
    try:
        while True:
            over_rules, wres = _game_over(st, ply, hist, max_plies)
            newly = ~done & over_rules
            result = torch.where(newly, wres, result)
            done = done | newly
            if bool(done.all()):
                break

            cand_turn = (st.stm == cand_white) & ~done
            sf_turn = (~cand_turn) & ~done
            code = torch.zeros(S, dtype=torch.int32, device=dev)

            idx = cand_turn.nonzero().squeeze(1)
            if idx.numel() > 0:
                res = M.search(net, st.index(idx), sims, c_puct=c_puct, add_noise=False, amp=amp)
                sel = res.visits.argmax(1)
                code[idx] = res.code.gather(1, sel[:, None]).squeeze(1)

            sidx = sf_turn.nonzero().squeeze(1).tolist()
            if sidx:
                boards_now = [G.board_from_state(st, i) for i in sidx]
                tasks = [(engines[k % sf_workers], b) for k, b in enumerate(boards_now)]
                moves = list(pool.map(sf_move, tasks))
                for i, b, mv in zip(sidx, boards_now, moves):
                    if mv is None:
                        legal = list(b.legal_moves)
                        mv = legal[0] if legal else None
                    if mv is not None:
                        code[i] = G.move_to_code(mv, b)

            adv = ~done
            new_st = G.make_moves(st, code)
            st.assign(adv, new_st)
            ply = ply + adv.long()
            nh = G.hash_state(st)
            pidx = ply.clamp(max=max_plies + 1)
            hist[ar, pidx] = torch.where(adv, nh, hist[ar, pidx])
    finally:
        for e in engines:
            try:
                e.quit()
            except Exception:
                pass
        pool.shutdown(wait=False)

    r = result.view(pairs, 2)   # [pair, (cand white, cand black)]
    cand_scores = torch.stack([(r[:, 0] + 1) / 2, (1 - r[:, 1]) / 2], dim=1).reshape(-1)
    wins = int((cand_scores == 1.0).sum())
    draws = int((cand_scores == 0.5).sum())
    losses = int((cand_scores == 0.0).sum())
    score = float(cand_scores.mean())
    n = S
    opp_elo = SKILL_ELO.get(skill_level, 1500)
    est = opp_elo + _elo_diff(score, n)
    return {
        "skill_level": skill_level, "opponent_elo": opp_elo, "games": n,
        "score": round(score, 3), "wins": wins, "draws": draws, "losses": losses,
        "estimated_elo": round(est), "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--skills", default="4,7,10", help="Comma-separated Stockfish Skill Level anchors")
    ap.add_argument("--pairs", type=int, default=40, help="Openings per skill level (2 games each)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--movetime", type=float, default=0.1)
    ap.add_argument("--sf-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH", "stockfish"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No GPU available; refusing to run on the CPU.")

    net, _ = load_checkpoint(args.candidate, device="cuda")
    skills = [int(s) for s in args.skills.split(",") if s]

    results = []
    for level in skills:
        print(f"--- skill {level} (anchor {SKILL_ELO.get(level, 1500)} Elo): "
              f"{2 * args.pairs} games, {args.sims} sims, {args.movetime}s/move ---", flush=True)
        r = run_match(
            net, args.stockfish, args.pairs, level, args.movetime, args.sims,
            args.sf_workers, args.seed + level,
        )
        results.append(r)
        print(f"  +{r['wins']} ={r['draws']} -{r['losses']}  score {r['score']:.3f}  "
              f"-> estimated Elo {r['estimated_elo']}  ({r['seconds']}s)", flush=True)

    print("\n=== summary ===", flush=True)
    for r in results:
        print(r, flush=True)
    avg = sum(r["estimated_elo"] for r in results) / len(results)
    print(f"\nmean estimated Elo across anchors: {avg:.0f}", flush=True)


if __name__ == "__main__":
    main()
