"""Leela Chess Zero teacher: (value, policy) targets from a GPU-backed network search.

Same interface as :class:`teacher.stockfish.StockfishTeacher`, so the labeling
pipeline can use either. lc0 searches by node count (not depth). The policy
target is the root visit distribution (AlphaZero-style), read from lc0's
``VerboseMoveStats`` output, and the value target is the root WDL expectation
``P(win) - P(loss)`` from the side to move's perspective, already in [-1, 1].

lc0 is driven over a minimal blocking UCI pipe instead of python-chess's async
engine wrapper: streaming ``analysis()`` there busy-spins ~2 CPU cores per engine
on Windows while waiting, which starves the other labeling workers.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Dict, Optional

import chess
import numpy as np

from engine.encoding import move_to_index

# "info string e2e4  (322 ) N:      12 (+ 0) (P: 20.50%) ..."  (N = root visits)
_MOVE_STATS = re.compile(r"^(\S+)\s+\(\s*\d+\s*\)\s+N:\s*(\d+)")


class Lc0Teacher:
    def __init__(
        self,
        path: str,
        weights: str,
        nodes: int = 400,
        backend: str = "onnx-dml",
        backend_opts: str = "",
        threads: int = 1,
        minibatch: int = 32,
        cache_size: int = 20000,
        policy_temp: float = 1.0,
    ):
        path, weights = os.path.abspath(path), os.path.abspath(weights)
        if not os.path.exists(path):
            raise FileNotFoundError(f"lc0 executable not found: {path}")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"lc0 weights not found: {weights}")
        self.path = path
        self.nodes = nodes
        self.policy_temp = policy_temp
        cmd = [path, f"--weights={weights}", f"--backend={backend}"]
        if backend_opts:
            cmd.append(f"--backend-opts={backend_opts}")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until(lambda line: line == "uciok")
        options = {
            "Threads": threads,
            "MinibatchSize": minibatch,
            "NNCacheSize": cache_size,
            "UCI_ShowWDL": "true",
            "VerboseMoveStats": "true",
        }
        for name, value in options.items():
            self._send(f"setoption name {name} value {value}")
        self._send("isready")  # blocks until the network is loaded
        self._read_until(lambda line: line == "readyok")

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, done, on_line=None) -> None:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"lc0 exited unexpectedly (code {self.proc.poll()})")
            line = line.strip()
            if on_line is not None:
                on_line(line)
            if done(line):
                return

    def label(self, board: chess.Board, policy_temp: Optional[float] = None) -> Dict[str, object]:
        temp = self.policy_temp if policy_temp is None else policy_temp
        legal = list(board.legal_moves)
        if not legal:
            return {
                "value": -1.0 if board.is_checkmate() else 0.0,
                "policy_indices": [],
                "policy_probs": [],
                "best_move": None,
            }

        visits: Dict[chess.Move, int] = {}
        state = {"value": 0.0, "best": None}

        def on_line(line: str) -> None:
            if line.startswith("info string "):
                m = _MOVE_STATS.match(line[len("info string "):])
                if m and m.group(1) != "node":
                    try:
                        move = chess.Move.from_uci(m.group(1))
                    except ValueError:
                        return
                    n = int(m.group(2))
                    if n > 0 and move in board.legal_moves:
                        visits[move] = n
            elif line.startswith("info "):
                tokens = line.split()
                if "wdl" in tokens:
                    i = tokens.index("wdl")
                    w, d, l = (int(t) for t in tokens[i + 1:i + 4])
                    state["value"] = (w - l) / max(w + d + l, 1)
                if "pv" in tokens:
                    try:
                        state["best"] = chess.Move.from_uci(tokens[tokens.index("pv") + 1])
                    except (ValueError, IndexError):
                        pass

        self._send(f"position fen {board.fen()}")
        self._send(f"go nodes {self.nodes}")
        self._read_until(lambda line: line.startswith("bestmove"), on_line)
        value, best_move = state["value"], state["best"]

        if not visits:
            # e.g. a single legal move can be returned without a full search.
            move = best_move or legal[0]
            visits = {move: 1}
        if best_move is None:
            best_move = max(visits, key=visits.get)

        moves = list(visits)
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        probs = counts ** (1.0 / max(temp, 1e-6))
        probs /= probs.sum()

        return {
            "value": float(value),
            "policy_indices": [move_to_index(m) for m in moves],
            "policy_probs": probs.tolist(),
            "best_move": best_move.uci(),
        }

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "Lc0Teacher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
