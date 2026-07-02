"""Board and move encoding.

Design choice: we use *absolute* board orientation (no flipping for the side to
move) and include an explicit "side to move" plane. This is slightly less sample
efficient than AlphaZero's perspective flipping, but it is dramatically simpler
and removes a whole class of coordinate bugs -- a good trade for a 1500-2000 Elo
target.

Input planes (18 total, each 8x8):
    0-5   : white P, N, B, R, Q, K
    6-11  : black P, N, B, R, Q, K
    12    : white kingside castling rights  (full plane of 1s if available)
    13    : white queenside castling rights
    14    : black kingside castling rights
    15    : black queenside castling rights
    16    : en-passant target square (single 1)
    17    : side to move (all 1s if white to move, else all 0s)

Move encoding: a fixed, deterministic vocabulary of (from_square, to_square,
promotion) tuples. Queen promotions collapse onto the plain (from, to) move
(there is never an ambiguous legal collision because only one piece occupies a
square). Under-promotions to knight/rook/bishop get dedicated indices.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import chess
import numpy as np

NUM_PLANES = 18

_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
_PIECE_TO_OFFSET = {pt: i for i, pt in enumerate(_PIECE_ORDER)}

MoveKey = Tuple[int, int, object]  # (from_square, to_square, promotion|None)


def _build_move_vocab() -> Tuple[Dict[MoveKey, int], List[MoveKey]]:
    move_to_idx: Dict[MoveKey, int] = {}
    idx_to_move: List[MoveKey] = []

    def add(key: MoveKey) -> None:
        if key not in move_to_idx:
            move_to_idx[key] = len(idx_to_move)
            idx_to_move.append(key)

    # All plain from/to moves (queen promotions fold into these).
    for frm in range(64):
        for to in range(64):
            if frm == to:
                continue
            add((frm, to, None))

    # Under-promotions (knight, rook, bishop) for both colors.
    underpromo = [chess.KNIGHT, chess.ROOK, chess.BISHOP]
    for color in (chess.WHITE, chess.BLACK):
        from_rank = 6 if color == chess.WHITE else 1
        to_rank = 7 if color == chess.WHITE else 0
        for file in range(8):
            frm = chess.square(file, from_rank)
            for dfile in (-1, 0, 1):
                tf = file + dfile
                if 0 <= tf < 8:
                    to = chess.square(tf, to_rank)
                    for promo in underpromo:
                        add((frm, to, promo))

    return move_to_idx, idx_to_move


MOVE_TO_IDX, IDX_TO_MOVE = _build_move_vocab()
POLICY_SIZE = len(IDX_TO_MOVE)


def move_to_index(move: chess.Move) -> int:
    """Map a python-chess Move to its policy index."""
    promo = move.promotion
    if promo is None or promo == chess.QUEEN:
        key: MoveKey = (move.from_square, move.to_square, None)
    else:
        key = (move.from_square, move.to_square, promo)
    return MOVE_TO_IDX[key]


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Best-effort decode of a policy index into a legal move on ``board``.

    Because the base (from, to) index is shared between a plain move and a queen
    promotion, we resolve the ambiguity using the board: if the moving piece is a
    pawn reaching the last rank we attach a queen promotion.
    """
    frm, to, promo = IDX_TO_MOVE[index]
    if promo is None:
        piece = board.piece_at(frm)
        to_rank = chess.square_rank(to)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promo = chess.QUEEN
    return chess.Move(frm, to, promotion=promo)


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a board into an (18, 8, 8) float32 tensor of planes."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        offset = _PIECE_TO_OFFSET[piece.piece_type]
        plane = offset if piece.color == chess.WHITE else 6 + offset
        planes[plane, rank, file] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12].fill(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13].fill(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14].fill(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15].fill(1.0)

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[16, rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[17].fill(1.0)

    return planes


def legal_move_indices(board: chess.Board) -> Tuple[List[chess.Move], List[int]]:
    """Return the legal moves and their corresponding policy indices."""
    moves = list(board.legal_moves)
    idxs = [move_to_index(m) for m in moves]
    return moves, idxs
