"""Generate diverse positions to be labeled by the teacher.

Strategy: play random games of randomized length and snapshot positions after
the opening phase. Using *uniform* random moves (rather than biasing toward
captures) keeps pieces on the board longer, which yields far more distinct
midgame positions and avoids the duplicate explosion you get when games collapse
into a small set of simplified endgames. A light de-duplication set removes exact
repeats, and a stall safeguard guarantees the function always terminates.

The teacher assigns correct value/policy targets regardless of how "natural" a
position is, so broad coverage is what matters for the supervised bootstrap.
"""
from __future__ import annotations

import random
from typing import List, Optional, Set

import chess


def _board_key(fen: str) -> str:
    # Placement + side to move + castling + ep (ignore move clocks).
    return " ".join(fen.split(" ")[:4])


def generate_positions(
    n: int,
    min_ply: int = 8,
    max_ply: int = 120,
    samples_per_game: int = 8,
    seed: Optional[int] = None,
    progress=None,
    progress_every: int = 2000,
    max_attempts_factor: int = 40,
) -> List[str]:
    """Return up to ``n`` unique FENs.

    ``progress`` is an optional callable(done, total) invoked roughly every
    ``progress_every`` positions. The routine stops early (returning fewer than
    ``n``) if it stalls finding new unique positions, so it never hangs.

    Snapshots are taken at the *start* of a ply (so the recorded position is
    guaranteed to have legal moves, i.e. is non-terminal) and the per-ply move
    list is reused for both the snapshot decision and the move that is played,
    which keeps generation fast.
    """
    rng = random.Random(seed)
    seen: Set[str] = set()
    out: List[str] = []

    # Include the start position for opening coverage.
    start = chess.Board().fen()
    seen.add(_board_key(start))
    out.append(start)

    last_reported = 0
    stall = 0
    stall_limit = max(10_000, n * max_attempts_factor)

    while len(out) < n and stall < stall_limit:
        board = chess.Board()
        game_len = rng.randint(min_ply, max_ply)
        # Ply indices at which to snapshot this game (after the opening).
        snap_points = {rng.randint(min_ply, game_len) for _ in range(samples_per_game)}

        for ply in range(1, game_len + 1):
            moves = list(board.legal_moves)
            if not moves:
                break
            if ply in snap_points:
                fen = board.fen()
                key = _board_key(fen)
                if key in seen:
                    stall += 1
                else:
                    seen.add(key)
                    out.append(fen)
                    stall = 0
                    if progress and len(out) - last_reported >= progress_every:
                        progress(len(out), n)
                        last_reported = len(out)
                    if len(out) >= n:
                        break
            board.push(rng.choice(moves))

    if progress:
        progress(len(out), n)
    return out[:n]
