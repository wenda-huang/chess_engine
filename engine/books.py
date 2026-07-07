"""Opening book (Polyglot) and endgame tablebase (Syzygy) helpers.

These are used for actual play, board-editor analysis, arena gating openings,
and ending games once Syzygy proves the result. Training labels still come from
Stockfish; self-play/arena use tablebases only to stop solved endgames early.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional

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


def try_load_opening_book(path: str) -> Optional[OpeningBook]:
    if path and os.path.exists(path):
        try:
            return OpeningBook(path)
        except Exception:
            return None
    return None


def try_load_tablebase(path: str) -> Optional[Tablebase]:
    if path and os.path.isdir(path):
        try:
            return Tablebase(path)
        except Exception:
            return None
    return None


def tablebase_forced_white_result(board: chess.Board, tb: Tablebase) -> Optional[float]:
    """If Syzygy proves the outcome, return white_result (1/0/-1). Else None.

    - WDL -2: side to move is lost -> immediate resignation
    - WDL  0: proven draw -> claim draw
    - WDL  2: winning for STM -> keep playing (opponent resigns on their -2 turn)
    """
    if not tb.available(board):
        return None
    wdl = tb.probe_wdl(board)
    if wdl is None:
        return None
    if wdl == -2:
        return -1.0 if board.turn == chess.WHITE else 1.0
    if wdl == 0:
        return 0.0
    return None


def sample_book_opening(
    book: OpeningBook,
    rng: random.Random,
    min_plies: int = 6,
    max_plies: int = 24,
    stop_prob: float = 0.12,
) -> Optional[List[chess.Move]]:
    """Walk the Polyglot book from the start position; stop when the book ends."""
    board = chess.Board()
    moves: List[chess.Move] = []
    for _ in range(max_plies):
        mv = book.move(board, weighted=True)
        if mv is None:
            break
        moves.append(mv)
        board.push(mv)
        if len(moves) >= min_plies and rng.random() < stop_prob:
            break
    return moves if len(moves) >= min_plies else None


def sample_random_opening(
    rng: random.Random,
    min_plies: int = 4,
    max_plies: int = 12,
) -> List[chess.Move]:
    """Uniform-random legal moves (legacy arena variety)."""
    board = chess.Board()
    opening: List[chess.Move] = []
    k = rng.randint(min_plies, max_plies)
    for _ in range(k):
        if board.is_game_over():
            break
        mv = rng.choice(list(board.legal_moves))
        opening.append(mv)
        board.push(mv)
    return opening


def build_mixed_opening_pool(
    rng: random.Random,
    size: int = 50,
    book: Optional[OpeningBook] = None,
    book_fraction: float = 0.7,
    min_plies: int = 6,
    max_plies: int = 24,
    random_min_plies: int = 4,
    random_max_plies: int = 12,
) -> List[List[chess.Move]]:
    """Build arena openings: mostly Polyglot book lines + some random lines."""
    pool: List[List[chess.Move]] = []
    seen: set[str] = set()

    def add_opening(moves: List[chess.Move]) -> bool:
        if not moves:
            return False
        key = " ".join(m.uci() for m in moves)
        if key in seen:
            return False
        seen.add(key)
        pool.append(moves)
        return True

    target_book = int(round(size * book_fraction)) if book is not None else 0
    attempts = 0
    max_attempts = max(size * 20, 200)

    while book is not None and len(pool) < target_book and attempts < max_attempts:
        attempts += 1
        sampled = sample_book_opening(
            book, rng, min_plies=min_plies, max_plies=max_plies,
        )
        if sampled is not None:
            add_opening(sampled)

    while len(pool) < size and attempts < max_attempts:
        attempts += 1
        if book is not None and rng.random() < book_fraction and len(pool) < target_book + 10:
            sampled = sample_book_opening(
                book, rng, min_plies=min_plies, max_plies=max_plies,
            )
            if sampled is not None and add_opening(sampled):
                continue
        add_opening(
            sample_random_opening(
                rng, min_plies=random_min_plies, max_plies=random_max_plies,
            )
        )

    return pool
