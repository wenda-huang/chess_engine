"""Strength evaluation: estimate Elo by playing matches vs skill-limited Stockfish.

Absolute Elo is only approximate -- we anchor Stockfish skill levels to rough Elo
values and derive our engine's Elo from the match score. It is reliable for
tracking *relative* progress between checkpoints.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional

import chess
import chess.engine

from engine.config import Config
from engine.player import EnginePlayer

# Rough anchor: Stockfish "Skill Level" UCI option -> approximate Elo.
# These are ballpark values from community testing at short time controls.
SKILL_ELO = {
    0: 1350,
    1: 1425,
    2: 1500,
    3: 1575,
    4: 1650,
    5: 1750,
    6: 1850,
    7: 1950,
    8: 2050,
    9: 2150,
    10: 2250,
    12: 2450,
    15: 2700,
    20: 3000,
}


def _score_to_elo_diff(score: float, n: int) -> float:
    # Clamp to avoid infinities on clean sweeps.
    eps = 1.0 / (2 * n)
    score = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


def play_game(
    player: EnginePlayer,
    opponent_move: Callable[[chess.Board], Optional[chess.Move]],
    player_is_white: bool,
    sims: int,
    max_moves: int = 240,
    random_opening_plies: int = 2,
    rng: Optional[random.Random] = None,
) -> float:
    """Play one game. Returns 1.0 (player win), 0.5 (draw), 0.0 (loss)."""
    rng = rng or random.Random()
    board = chess.Board()

    # A couple of random opening plies for variety.
    for _ in range(random_opening_plies):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))

    use_books = getattr(player, "opening_book", None) is not None or \
        getattr(player, "tablebase", None) is not None
    while not board.is_game_over(claim_draw=True) and board.fullmove_number < max_moves:
        player_to_move = board.turn == (chess.WHITE if player_is_white else chess.BLACK)
        if player_to_move:
            if use_books:
                move, _ = player.play_move(board, simulations=sims)
            else:
                move, _ = player.select_move(board, simulations=sims, temperature=0.0)
        else:
            move = opponent_move(board)
            if move is None:
                break
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    won = outcome.winner == (chess.WHITE if player_is_white else chess.BLACK)
    return 1.0 if won else 0.0


def estimate_elo(
    player: EnginePlayer,
    config: Optional[Config] = None,
    games: int = 20,
    sims: int = 80,
    skill_level: int = 3,
    movetime: float = 0.05,
    progress=None,
) -> dict:
    """Play ``games`` vs Stockfish at ``skill_level`` and estimate Elo."""
    config = config or Config()
    rng = random.Random(1234)

    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    try:
        try:
            engine.configure({"Skill Level": skill_level})
        except Exception:
            pass

        def opponent_move(board: chess.Board) -> Optional[chess.Move]:
            result = engine.play(board, chess.engine.Limit(time=movetime))
            return result.move

        total = 0.0
        wins = draws = losses = 0
        for g in range(games):
            player_is_white = g % 2 == 0
            r = play_game(player, opponent_move, player_is_white, sims, rng=rng)
            total += r
            if r == 1.0:
                wins += 1
            elif r == 0.5:
                draws += 1
            else:
                losses += 1
            if progress:
                progress(
                    {
                        "event": "eval_game",
                        "game": g + 1,
                        "games": games,
                        "result": r,
                        "score": round(total, 1),
                    }
                )
    finally:
        engine.quit()

    score = total / games
    opp_elo = SKILL_ELO.get(skill_level, 1500)
    est = opp_elo + _score_to_elo_diff(score, games)
    result = {
        "event": "eval_done",
        "games": games,
        "score": round(score, 3),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "opponent_skill": skill_level,
        "opponent_elo": opp_elo,
        "estimated_elo": round(est),
    }
    if progress:
        progress(result)
    return result
