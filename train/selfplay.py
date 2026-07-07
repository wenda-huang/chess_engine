"""Phase 2: self-play game generation for AlphaZero-style refinement."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import chess
import numpy as np

from engine.config import Config
from engine.encoding import board_to_planes
from engine.books import Tablebase, tablebase_forced_white_result
from engine.player import EnginePlayer
from teacher.stockfish import StockfishTeacher, cp_to_value

Sample = Tuple[np.ndarray, np.ndarray, float]


def play_selfplay_game(
    player: EnginePlayer,
    sims: int,
    temperature_moves: int = 25,
    max_moves: int = 200,
    teacher: Optional[StockfishTeacher] = None,
    sf_value_weight: float = 0.0,
    rng: Optional[random.Random] = None,
    resign: bool = True,
    resign_threshold: float = 0.95,
    resign_streak: int = 3,
    complete_fraction: float = 0.08,
    tablebase: Optional[Tablebase] = None,
) -> Tuple[List[Sample], float]:
    """Play one self-play game; return (samples, white_result).

    Each sample is (planes, visit-count policy, value_target). The value target
    is the game outcome from that position's side-to-move perspective, optionally
    blended with the Stockfish eval at that position (``sf_value_weight``).
    ``white_result`` is 1.0 (white win), 0.0 (draw) or -1.0 (black win).

    Resignation (AlphaZero-style): if the side to move sees value below
    ``-resign_threshold`` for ``resign_streak`` consecutive moves, the game ends.
    A fraction ``complete_fraction`` of games always play to completion to avoid
    resignation bias in the training data.

    Tablebase: if Syzygy proves a loss for the side to move (WDL -2) or a draw
    (WDL 0), the game ends immediately without further search.
    """
    rng = rng or random.Random()
    board = chess.Board()
    history: List[Tuple[np.ndarray, np.ndarray, chess.Color, Optional[float]]] = []
    play_to_completion = (not resign) or rng.random() < complete_fraction
    bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
    white_result: Optional[float] = None
    winner: Optional[chess.Color] = None

    move_number = 0
    while not board.is_game_over(claim_draw=True) and move_number < max_moves:
        if tablebase is not None:
            wr = tablebase_forced_white_result(board, tablebase)
            if wr is not None:
                white_result = wr
                winner = None if wr == 0.0 else (chess.WHITE if wr > 0 else chess.BLACK)
                break

        temperature = 1.0 if move_number < temperature_moves else 0.0
        move, root = player.select_move(
            board, simulations=sims, temperature=temperature, add_noise=True
        )
        planes = board_to_planes(board)
        pi = player.policy_target(root)

        sf_value = None
        if teacher is not None and sf_value_weight > 0.0:
            info = teacher.analyse(board)
            score = info[0]["score"].pov(board.turn)
            sf_value = cp_to_value(score.score(), score.mate())

        history.append((planes, pi, board.turn, sf_value))

        if resign and not play_to_completion:
            q = root.Q if root.N > 0 else 0.0
            stm = board.turn
            if q < -resign_threshold:
                bad_streak[stm] += 1
            else:
                bad_streak[stm] = 0
            if bad_streak[stm] >= resign_streak:
                winner = chess.BLACK if stm == chess.WHITE else chess.WHITE
                white_result = 1.0 if winner == chess.WHITE else -1.0
                break

        board.push(move)
        move_number += 1

    if white_result is None:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            winner = None
            white_result = 0.0
        else:
            winner = outcome.winner
            white_result = 1.0 if winner == chess.WHITE else -1.0

    samples: List[Sample] = []
    for planes, pi, color, sf_value in history:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == color else -1.0
        if sf_value is not None:
            z = (1 - sf_value_weight) * z + sf_value_weight * sf_value
        samples.append((planes, pi, float(z)))
    return samples, white_result
