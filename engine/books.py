"""Opening book (Polyglot) and endgame tablebase (Syzygy) helpers.

These are used only for actual play and board-editor analysis -- they give a
small net perfect opening theory and perfect low-piece endgames, which is where
it is weakest. Training and Elo evaluation deliberately do NOT use them, so we
keep measuring the raw network.

Both wrappers are best-effort: if the files are missing or a probe fails, the
methods return ``None`` and the caller falls back to the neural net.
"""
from __future__ import annotations

from typing import Optional

import chess
import chess.polyglot
import chess.syzygy


class OpeningBook:
    """Thin wrapper over a Polyglot ``.bin`` opening book."""

    def __init__(self, path: str):
        self.reader = chess.polyglot.open_reader(path)

    def move(self, board: chess.Board, weighted: bool = True) -> Optional[chess.Move]:
        """Return a book move for ``board`` (weighted by popularity), or None."""
        try:
            if weighted:
                entry = self.reader.weighted_choice(board)
            else:
                entry = self.reader.find(board)
            move = entry.move
            return move if move in board.legal_moves else None
        except (IndexError, KeyError):
            return None  # position not in book
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass


class Tablebase:
    """Thin wrapper over Syzygy WDL/DTZ tablebases."""

    def __init__(self, path: str, max_pieces: int = 7):
        self.tb = chess.syzygy.open_tablebase(path)
        self.max_pieces = max_pieces

    def available(self, board: chess.Board) -> bool:
        # Tablebases don't cover positions with castling rights.
        if board.castling_rights:
            return False
        return chess.popcount(board.occupied) <= self.max_pieces

    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        """WDL from the side-to-move perspective (2 win .. -2 loss), or None."""
        try:
            return self.tb.probe_wdl(board)
        except Exception:
            return None

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Pick a WDL-optimal move (DTZ tiebreak). None if not fully covered."""
        if not self.available(board):
            return None
        scored = []
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = self.tb.probe_wdl(board)  # opponent's perspective after our move
            except Exception:
                board.pop()
                return None  # incomplete coverage -> defer to the net
            try:
                dtz = self.tb.probe_dtz(board)
            except Exception:
                dtz = 0
            board.pop()
            scored.append((move, -wdl, dtz))  # our result is the negation
        if not scored:
            return None
        best_wdl = max(s[1] for s in scored)
        cands = [s for s in scored if s[1] == best_wdl]
        if best_wdl > 0:
            # Winning: convert fastest (opponent DTZ closest to zero).
            cands.sort(key=lambda s: s[2], reverse=True)
        elif best_wdl < 0:
            # Losing: drag it out.
            cands.sort(key=lambda s: s[2])
        return cands[0][0]

    def close(self) -> None:
        try:
            self.tb.close()
        except Exception:
            pass
