"""Phase 2: self-play game generation for AlphaZero-style refinement."""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import chess
import numpy as np

from engine.config import Config
from engine.encoding import board_to_planes
from engine.player import EnginePlayer
from teacher.stockfish import StockfishTeacher, cp_to_value

Sample = Tuple[np.ndarray, np.ndarray, float]


def play_selfplay_game(
    player: EnginePlayer,
    sims: int,
    temperature_moves: int = 20,
    max_moves: int = 200,
    teacher: Optional[StockfishTeacher] = None,
    sf_value_weight: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Sample], float]:
    """Play one self-play game; return (samples, white_result).

    Each sample is (planes, visit-count policy, value_target). The value target
    is the game outcome from that position's side-to-move perspective, optionally
    blended with the Stockfish eval at that position (``sf_value_weight``).
    ``white_result`` is 1.0 (white win), 0.0 (draw) or -1.0 (black win).
    """
    rng = rng or random.Random()
    board = chess.Board()
    history: List[Tuple[np.ndarray, np.ndarray, chess.Color, Optional[float]]] = []

    move_number = 0
    while not board.is_game_over(claim_draw=True) and move_number < max_moves:
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
        board.push(move)
        move_number += 1

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
