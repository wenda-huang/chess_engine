#!/usr/bin/env python3
"""Head-to-head match between two checkpoints, fully on the GPU (``engine.gpuplay.gpu_arena``).

Each opening is played twice with colours swapped; the score and Elo difference are
from the candidate's point of view.

    python scripts/arena.py models/distill_small.pt models/best_small_it197.pt --pairs 150 --sims 400
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

# hipBLASLt cannot run under HIP graph capture; must be set before torch loads.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine.books import build_mixed_opening_pool, try_load_opening_book
from engine.config import Config
from engine.gpuplay import gpu_arena
from engine.model import load_checkpoint


def _elo(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("champion")
    ap.add_argument("--pairs", type=int, default=150, help="Openings (2 games each)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No GPU available; refusing to run the arena on the CPU.")
    cand, _ = load_checkpoint(args.candidate, device="cuda")
    champ, _ = load_checkpoint(args.champion, device="cuda")
    book = try_load_opening_book(Config().opening_book_path)
    openings = build_mixed_opening_pool(random.Random(args.seed), size=args.pairs, book=book)

    print(f"playing {2 * len(openings)} games at {args.sims} sims (results arrive at the end)", flush=True)
    t0 = time.time()
    score, w, d, l = gpu_arena(cand, champ, openings, sims=args.sims, amp=True)
    n = w + d + l
    # 95% interval from the per-game score variance.
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * score ** 2) / max(n - 1, 1)
    half = 1.96 * math.sqrt(var / max(n, 1))
    print(f"{os.path.basename(args.candidate)} vs {os.path.basename(args.champion)}: "
          f"+{w} ={d} -{l}  score {score:.3f}  Elo {_elo(score):+.0f} "
          f"[{_elo(score - half):+.0f}, {_elo(score + half):+.0f}]  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
