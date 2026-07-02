"""Stockfish teacher: turns a position into (value, policy) supervision targets.

Uses python-chess's UCI bridge. A single :class:`StockfishTeacher` owns one
engine process; create it once and reuse it for many positions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import chess
import chess.engine
import numpy as np

from engine.encoding import POLICY_SIZE, move_to_index

# Centipawn -> value scaling. value = tanh(cp / CP_SCALE); ~+/-0.9 around 400cp.
CP_SCALE = 350.0
MATE_VALUE = 1.0


def cp_to_value(cp: Optional[int], mate: Optional[int]) -> float:
    """Convert a score (from the mover's perspective) to a value in [-1, 1]."""
    if mate is not None:
        return MATE_VALUE if mate > 0 else -MATE_VALUE
    if cp is None:
        return 0.0
    return math.tanh(cp / CP_SCALE)


class StockfishTeacher:
    def __init__(
        self,
        path: str,
        depth: int = 12,
        movetime: Optional[float] = None,
        multipv: int = 4,
        threads: int = 1,
        hash_mb: int = 64,
    ):
        self.path = path
        self.depth = depth
        self.movetime = movetime
        self.multipv = multipv
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Threads": threads, "Hash": hash_mb})
        except Exception:
            # Some builds reject unknown options; ignore.
            pass

    def _limit(self) -> chess.engine.Limit:
        if self.movetime is not None:
            return chess.engine.Limit(time=self.movetime)
        return chess.engine.Limit(depth=self.depth)

    def analyse(self, board: chess.Board) -> List[dict]:
        """Return raw MultiPV analysis info dicts."""
        info = self.engine.analyse(board, self._limit(), multipv=self.multipv)
        if isinstance(info, dict):
            info = [info]
        return info

    def label(self, board: chess.Board, policy_temp: float = 0.5) -> Dict[str, object]:
        """Produce a training label for ``board``.

        Returns a dict with:
            value          : float in [-1, 1], from side-to-move perspective
            policy_indices : list[int]   (policy vocab indices)
            policy_probs   : list[float] (sum to 1)
            best_move      : uci string
        """
        info = self.analyse(board)
        # Best line first.
        pov = board.turn
        scores: List[float] = []
        indices: List[int] = []
        best_move = None

        for i, entry in enumerate(info):
            pv = entry.get("pv")
            if not pv:
                continue
            move = pv[0]
            score = entry["score"].pov(pov)
            cp = score.score(mate_score=100000)
            scores.append(float(cp))
            indices.append(move_to_index(move))
            if i == 0:
                best_move = move

        if not indices:
            # No legal moves analysed (terminal); fall back to uniform over legal.
            legal = list(board.legal_moves)
            if not legal:
                return {
                    "value": self._terminal_value(board),
                    "policy_indices": [],
                    "policy_probs": [],
                    "best_move": None,
                }
            indices = [move_to_index(m) for m in legal]
            scores = [0.0] * len(indices)
            best_move = legal[0]

        # Value from the best line (mover's perspective).
        top = info[0]["score"].pov(pov)
        value = cp_to_value(top.score(), top.mate())

        # Policy target: softmax over the MultiPV centipawn scores.
        probs = _softmax(np.array(scores, dtype=np.float64) / max(CP_SCALE * policy_temp, 1e-6))

        return {
            "value": float(value),
            "policy_indices": indices,
            "policy_probs": probs.tolist(),
            "best_move": best_move.uci() if best_move else None,
        }

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        result = self.engine.play(board, self._limit())
        return result.move

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        return 0.0

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "StockfishTeacher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def dense_policy(indices: List[int], probs: List[float]) -> np.ndarray:
    """Expand sparse (indices, probs) into a dense POLICY_SIZE vector."""
    vec = np.zeros(POLICY_SIZE, dtype=np.float32)
    for i, p in zip(indices, probs):
        vec[i] = p
    return vec
