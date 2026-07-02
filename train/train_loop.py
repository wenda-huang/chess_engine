"""The AlphaZero refinement loop: self-play -> train -> evaluate -> promote."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from data.dataset import ReplayBuffer
from engine.config import Config
from engine.model import build_model, load_checkpoint, save_checkpoint
from engine.player import EnginePlayer
from teacher.stockfish import StockfishTeacher
from train.evaluate import estimate_elo
from train.progress import ProgressLogger
from train.selfplay import play_selfplay_game
from train.supervised import policy_loss


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
    player = EnginePlayer(model, config)

    teacher = None
    if sf_value_weight > 0.0:
        teacher = StockfishTeacher(config.stockfish_path, depth=8, multipv=1)

    out_path = os.path.join(config.models_dir, out_name)
    best_elo = -1e9

    progress.log(
        {
            "event": "start",
            "mode": "selfplay",
            "iterations": iterations,
            "games_per_iter": games_per_iter,
            "sims": sims,
            "device": device,
        }
    )

    try:
        for it in range(iterations):
            model.eval()
            new_samples = 0
            for g in range(games_per_iter):
                samples = play_selfplay_game(
                    player,
                    sims=sims,
                    teacher=teacher,
                    sf_value_weight=sf_value_weight,
                )
                buffer.add_game(samples)
                new_samples += len(samples)
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

            # Train on the replay buffer.
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

    progress.log({"event": "done", "mode": "selfplay", "checkpoint": out_path, "best_elo": best_elo})
    return out_path
