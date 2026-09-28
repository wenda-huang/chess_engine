#!/usr/bin/env python3
"""Dump the positions of GPU self-play games as FENs (start positions for lc0 labeling).

Search runs entirely on the GPU (``engine.gpuplay.gpu_selfplay``); finished games are
replayed on a python-chess board to recover each position. Duplicate positions (mostly
openings) are written once. Appends to ``--out`` and counts existing lines, so a
restarted run continues toward ``--n``.

    python scripts/selfplay_fens.py --checkpoint models/best_small_it197.pt --n 1900000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# hipBLASLt cannot run under HIP graph capture; must be set before torch loads.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import torch

from engine import gpuchess as G
from engine.gpuplay import GpuReplayBuffer, gpu_selfplay
from engine.model import load_checkpoint


def _key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])  # ignore move counters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_small.pt")
    ap.add_argument("--out", default="data_distill/fens.txt")
    ap.add_argument("--n", type=int, default=1_900_000, help="Unique positions to collect")
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--temperature-moves", type=int, default=30)
    ap.add_argument("--games-per-batch", type=int, default=512)
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--min-ply", type=int, default=4, help="Skip the first plies (book-like openings)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No GPU available; refusing to run self-play on the CPU.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    seen: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                seen.add(_key(chess.Board(line.strip())))
    print(f"resuming with {len(seen)} positions", flush=True)

    net, _ = load_checkpoint(args.checkpoint, device="cuda")
    buffer = GpuReplayBuffer(4096, "cuda")  # required by gpu_selfplay; contents unused
    out = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    start_n = len(seen)

    def on_moves(codes) -> None:
        board = chess.Board()
        for ply, code in enumerate(codes):
            if ply >= args.min_ply and not board.is_game_over():
                k = _key(board)
                if k not in seen:
                    seen.add(k)
                    out.write(board.fen() + "\n")
            move = G.code_to_move(code)
            piece = board.piece_type_at(move.from_square)
            if piece == chess.PAWN and chess.square_rank(move.to_square) in (0, 7) and not move.promotion:
                move.promotion = chess.QUEEN  # the GPU encoding stores queen promotions as 0
            if move not in board.legal_moves:
                break  # a resigned game's last code was never played
            board.push(move)

    batch = 0
    while len(seen) < args.n:
        stats = gpu_selfplay(net, buffer, args.games_per_batch, slots=args.slots, sims=args.sims,
                             temperature_moves=args.temperature_moves, amp=True, on_moves=on_moves)
        out.flush()
        batch += 1
        rate = (len(seen) - start_n) / max(time.time() - t0, 1e-9)
        eta = (args.n - len(seen)) / max(rate, 1e-9)
        print(f"batch {batch}: {len(seen)}/{args.n} positions  {rate:.0f}/s  "
              f"W{stats['white']:.0f} B{stats['black']:.0f} D{stats['draw']:.0f}  "
              f"eta {eta / 3600:.1f}h", flush=True)
    out.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
