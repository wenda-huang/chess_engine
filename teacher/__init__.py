"""Teachers that turn a position into (value, policy) supervision targets."""
from __future__ import annotations

from typing import Optional


def make_teacher(
    kind: str,
    config,
    depth: int = 12,
    movetime: Optional[float] = None,
    multipv: int = 4,
    threads: int = 1,
    hash_mb: int = 128,
    nodes: int = 400,
    minibatch: int = 32,
):
    """Build a teacher: ``"stockfish"`` (CPU, depth-limited) or ``"lc0"`` (GPU, node-limited)."""
    if kind == "lc0":
        from teacher.lc0 import Lc0Teacher

        return Lc0Teacher(
            config.lc0_path,
            config.lc0_weights,
            nodes=nodes,
            backend=config.lc0_backend,
            backend_opts=config.lc0_backend_opts,
            threads=threads,
            minibatch=minibatch,
        )
    if kind == "stockfish":
        from teacher.stockfish import StockfishTeacher

        return StockfishTeacher(
            config.stockfish_path, depth=depth, movetime=movetime,
            multipv=multipv, threads=threads, hash_mb=hash_mb,
        )
    raise ValueError(f"unknown teacher '{kind}' (expected 'stockfish' or 'lc0')")
