"""The AlphaZero refinement loop: self-play -> train -> evaluate -> promote.

Self-play can run across multiple worker processes. Each worker loads the
current network weights (refreshed every iteration) and generates games on its
own CPU core, while the main process performs the (GPU) training step. This is
the same parallelism trick used for labeling and gives a near-linear speedup in
games/hour on a many-core box.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import ReplayBuffer
from engine.config import Config, ModelConfig
from engine.model import build_model, load_checkpoint, save_checkpoint
from engine.player import EnginePlayer
from teacher.stockfish import StockfishTeacher
from train.evaluate import estimate_elo
from train.progress import ProgressLogger
from train.selfplay import play_selfplay_game
from train.supervised import policy_loss


# --------------------------------------------------------------------------- #
# Parallel self-play workers
# --------------------------------------------------------------------------- #
_SP: dict = {}


def _sp_init(model_config: dict, config: Config, sp_device: str, sf_value_weight: float) -> None:
    # Each worker is one of many; pin it to a single CPU thread so N workers don't
    # each spawn N intra-op threads and thrash the cores (crippling on CPU, and
    # still worth doing for the Python-side MCTS overhead when running on GPU).
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    config.device = sp_device
    _SP["model_config"] = model_config
    _SP["config"] = config
    _SP["device"] = sp_device
    _SP["version"] = None
    _SP["player"] = None
    _SP["sf_value_weight"] = sf_value_weight
    _SP["teacher"] = (
        StockfishTeacher(config.stockfish_path, depth=8, multipv=1)
        if sf_value_weight > 0.0
        else None
    )


def _sp_play(task):
    import random as _random

    version, weights_path, sims, temperature_moves, seed = task
    # Reload weights only when the model version changes (once per iteration).
    if _SP["version"] != version:
        model = build_model(ModelConfig(**_SP["model_config"]), device=_SP["device"])
        ckpt = torch.load(weights_path, map_location=_SP["device"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _SP["player"] = EnginePlayer(model, _SP["config"])
        _SP["version"] = version

    np.random.seed(seed % (2 ** 32))
    rng = _random.Random(seed)
    samples = play_selfplay_game(
        _SP["player"],
        sims=sims,
        temperature_moves=temperature_moves,
        teacher=_SP["teacher"],
        sf_value_weight=_SP["sf_value_weight"],
        rng=rng,
    )
    # Compress planes to uint8 (they are all 0/1) for cheaper inter-process transfer.
    return [(p.astype(np.uint8), pi, float(z)) for (p, pi, z) in samples]


def train_selfplay(
    config: Optional[Config] = None,
    iterations: int = 20,
    games_per_iter: int = 10,
    sims: int = 100,
    train_steps: int = 200,
    batch_size: int = 128,
    lr: float = 5e-4,
    buffer_capacity: int = 50_000,
    init_checkpoint: Optional[str] = "models/supervised.pt",
    out_name: str = "selfplay.pt",
    sf_value_weight: float = 0.0,
    temperature_moves: int = 20,
    workers: int = 1,
    selfplay_device: Optional[str] = None,
    eval_every: int = 5,
    eval_games: int = 12,
    eval_skill: int = 3,
    progress: Optional[ProgressLogger] = None,
) -> str:
    config = config or Config()
    config.ensure_dirs()
    progress = progress or ProgressLogger(os.path.join(config.logs_dir, "selfplay.jsonl"))
    device = config.device

    if init_checkpoint and os.path.exists(init_checkpoint):
        model, _ = load_checkpoint(init_checkpoint, device=device)
    else:
        model = build_model(config.model, device=device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    player = EnginePlayer(model, config)  # used for the sequential path and evaluation

    # Self-play workers default to CPU: it uses the spare cores, avoids GPU
    # oversubscription, and sidesteps CUDA-in-subprocess pitfalls.
    parallel = workers and workers > 1
    sp_device = selfplay_device or ("cpu" if parallel else device)

    teacher = None
    if sf_value_weight > 0.0 and not parallel:
        teacher = StockfishTeacher(config.stockfish_path, depth=8, multipv=1)

    pool = None
    if parallel:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=workers,
            initializer=_sp_init,
            initargs=(vars(model.config), config, sp_device, sf_value_weight),
        )

    out_path = os.path.join(config.models_dir, out_name)
    weights_path = os.path.join(config.models_dir, "_sp_weights.pt")
    best_elo = -1e9

    progress.log(
        {
            "event": "start",
            "mode": "selfplay",
            "iterations": iterations,
            "games_per_iter": games_per_iter,
            "sims": sims,
            "device": device,
            "workers": workers,
            "selfplay_device": sp_device,
        }
    )

    try:
        for it in range(iterations):
            model.eval()

            if parallel:
                # Publish current weights, then fan out games to the workers.
                save_checkpoint(weights_path, model)
                tasks = [
                    (it, weights_path, sims, temperature_moves, it * 100_003 + g)
                    for g in range(games_per_iter)
                ]
                done = 0
                for samples in pool.imap_unordered(_sp_play, tasks):
                    buffer.add_game(samples)
                    done += 1
                    progress.log(
                        {
                            "event": "selfplay_game",
                            "iter": it,
                            "game": done,
                            "games_per_iter": games_per_iter,
                            "samples": len(samples),
                            "buffer": len(buffer),
                        }
                    )
            else:
                for g in range(games_per_iter):
                    samples = play_selfplay_game(
                        player,
                        sims=sims,
                        temperature_moves=temperature_moves,
                        teacher=teacher,
                        sf_value_weight=sf_value_weight,
                    )
                    buffer.add_game(samples)
                    progress.log(
                        {
                            "event": "selfplay_game",
                            "iter": it,
                            "game": g + 1,
                            "games_per_iter": games_per_iter,
                            "samples": len(samples),
                            "buffer": len(buffer),
                        }
                    )

            # Train on the replay buffer (on the main/GPU device).
            model.train()
            if len(buffer) >= batch_size:
                for step in range(train_steps):
                    planes, target_policy, target_value = buffer.sample_batch(batch_size)
                    planes = planes.to(device)
                    target_policy = target_policy.to(device)
                    target_value = target_value.to(device)

                    logits, value = model(planes)
                    p_loss = policy_loss(logits, target_policy)
                    v_loss = F.mse_loss(value, target_value)
                    loss = p_loss + v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % 25 == 0:
                        progress.log(
                            {
                                "event": "train_step",
                                "iter": it,
                                "step": step,
                                "policy_loss": round(p_loss.item(), 4),
                                "value_loss": round(v_loss.item(), 4),
                            }
                        )

            save_checkpoint(out_path, model, meta={"iter": it})

            if eval_every and (it + 1) % eval_every == 0:
                model.eval()
                result = estimate_elo(
                    player,
                    config,
                    games=eval_games,
                    sims=max(40, sims // 2),
                    skill_level=eval_skill,
                    progress=progress,
                )
                elo = result["estimated_elo"]
                if elo > best_elo:
                    best_elo = elo
                    save_checkpoint(
                        os.path.join(config.models_dir, "best.pt"),
                        model,
                        meta={"iter": it, "estimated_elo": elo},
                    )
                    progress.log({"event": "promote", "iter": it, "estimated_elo": elo})
    finally:
        if teacher is not None:
            teacher.close()
        if pool is not None:
            pool.close()
            pool.join()

    progress.log({"event": "done", "mode": "selfplay", "checkpoint": out_path, "best_elo": best_elo})
    return out_path
