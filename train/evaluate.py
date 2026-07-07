"""Strength evaluation: estimate Elo by playing matches vs skill-limited Stockfish.

Absolute Elo is only approximate -- we anchor Stockfish skill levels to rough Elo
values and derive our engine's Elo from the match score. It is reliable for
tracking *relative* progress between checkpoints.
"""
from __future__ import annotations

import math
import os
import random
import time
from multiprocessing import Pool
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

_eval_worker: dict = {}


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


def _init_eval_worker(worker_cfg: dict) -> None:
    """Load one EnginePlayer + Stockfish per worker process."""
    global _eval_worker
    for key, value in worker_cfg.get("env", {}).items():
        if value:
            os.environ[key] = value

    config = Config()
    config.stockfish_path = worker_cfg["stockfish_path"]
    config.opening_book_path = worker_cfg.get("opening_book", "")
    config.syzygy_path = worker_cfg.get("syzygy", "")

    player = EnginePlayer(
        config=config,
        checkpoint=worker_cfg.get("checkpoint"),
        use_books=worker_cfg.get("use_books", False),
    )
    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
    try:
        engine.configure({"Skill Level": worker_cfg["skill_level"]})
    except Exception:
        pass

    _eval_worker = {
        "player": player,
        "engine": engine,
        "movetime": worker_cfg["movetime"],
    }


def _play_eval_game(task: tuple) -> dict:
    """Worker entry: play one game and return score + timing."""
    game_idx, player_is_white, sims, seed = task
    player = _eval_worker["player"]
    engine = _eval_worker["engine"]
    movetime = _eval_worker["movetime"]

    def opponent_move(board: chess.Board) -> Optional[chess.Move]:
        result = engine.play(board, chess.engine.Limit(time=movetime))
        return result.move

    t0 = time.time()
    rng = random.Random(seed)
    result = play_game(player, opponent_move, player_is_white, sims, rng=rng)
    return {
        "game": game_idx,
        "result": result,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _worker_config_from(config: Config, checkpoint: Optional[str], use_books: bool) -> dict:
    env = {}
    for key in ("CHESSAI_INFER", "CHESSAI_ONNX", "CHESSAI_DEVICE", "CHESSAI_COMPILE"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return {
        "checkpoint": checkpoint,
        "use_books": use_books,
        "stockfish_path": config.stockfish_path,
        "opening_book": config.opening_book_path,
        "syzygy": config.syzygy_path,
        "env": env,
    }


def estimate_elo(
    player: Optional[EnginePlayer] = None,
    config: Optional[Config] = None,
    games: int = 20,
    sims: int = 80,
    skill_level: int = 3,
    movetime: float = 0.05,
    progress=None,
    workers: int = 1,
    checkpoint: Optional[str] = None,
    use_books: bool = False,
) -> dict:
    """Play ``games`` vs Stockfish at ``skill_level`` and estimate Elo."""
    config = config or Config()
    workers = max(1, workers)

    if workers > 1:
        return _estimate_elo_parallel(
            config=config,
            games=games,
            sims=sims,
            skill_level=skill_level,
            movetime=movetime,
            progress=progress,
            workers=workers,
            checkpoint=checkpoint,
            use_books=use_books,
        )

    assert player is not None
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
            t0 = time.time()
            r = play_game(player, opponent_move, player_is_white, sims, rng=rng)
            elapsed = time.time() - t0
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
                        "elapsed_s": round(elapsed, 1),
                    }
                )
    finally:
        engine.quit()

    return _finalize_elo(games, total, wins, draws, losses, skill_level, progress)


def _estimate_elo_parallel(
    config: Config,
    games: int,
    sims: int,
    skill_level: int,
    movetime: float,
    progress,
    workers: int,
    checkpoint: Optional[str],
    use_books: bool,
) -> dict:
    worker_cfg = _worker_config_from(config, checkpoint, use_books)
    worker_cfg["skill_level"] = skill_level
    worker_cfg["movetime"] = movetime

    tasks = [
        (g, g % 2 == 0, sims, 1234 + g)
        for g in range(games)
    ]

    total = 0.0
    wins = draws = losses = 0
    done = 0

    with Pool(processes=workers, initializer=_init_eval_worker, initargs=(worker_cfg,)) as pool:
        for ev in pool.imap_unordered(_play_eval_game, tasks):
            done += 1
            r = ev["result"]
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
                        "game": done,
                        "games": games,
                        "result": r,
                        "score": round(total, 1),
                        "elapsed_s": ev["elapsed_s"],
                    }
                )

    return _finalize_elo(games, total, wins, draws, losses, skill_level, progress)


def _finalize_elo(
    games: int,
    total: float,
    wins: int,
    draws: int,
    losses: int,
    skill_level: int,
    progress,
) -> dict:
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
