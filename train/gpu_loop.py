"""Arena-gated self-play loop where game generation, search, replay and arena all run on the GPU.

Same algorithm and log events as ``train.train_loop.train_selfplay`` (champion/candidate split,
supervised anchor in every batch, arena gating), but there are no worker processes: games are
played in one batched search over hundreds of concurrent games (see ``engine.gpuplay``).
"""
from __future__ import annotations

import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import StockfishDataset
from engine.books import build_mixed_opening_pool, try_load_opening_book
from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.gpuplay import GpuReplayBuffer, gpu_arena, gpu_selfplay
from engine.model import build_model, load_checkpoint, save_checkpoint
from train.progress import ProgressLogger
from train.supervised import policy_loss
from train.train_loop import _gate_promote


class GpuAnchor:
    """A random subset of the labeled corpus held on the GPU (no CPU work per training step)."""

    def __init__(self, dataset: StockfishDataset, n: int, device, seed: int = 0):
        n = min(n, len(dataset))
        idx = np.sort(np.random.RandomState(seed).choice(len(dataset), size=n, replace=False))
        lens = dataset.pol_off[idx + 1] - dataset.pol_off[idx]
        K = max(int(lens.max()), 1)
        self.planes = torch.empty(n, 18, 8, 8, dtype=torch.uint8, device=device)
        self.pol = torch.zeros(n, K, dtype=torch.int16, device=device)
        self.prob = torch.zeros(n, K, dtype=torch.float16, device=device)
        self.value = torch.empty(n, dtype=torch.float32, device=device)
        for s in range(0, n, 50_000):
            planes, pi, pp, v = dataset.batch(idx[s:s + 50_000])
            e = s + len(v)
            self.planes[s:e] = torch.from_numpy(planes).to(device)
            self.pol[s:e, : pi.shape[1]] = torch.from_numpy(pi.astype(np.int16)).to(device)
            self.prob[s:e, : pp.shape[1]] = torch.from_numpy(pp).to(device).half()
            self.value[s:e] = torch.from_numpy(v).to(device)
        self.n = n
        self.device = device

    def sample_batch(self, b: int):
        i = torch.randint(0, self.n, (b,), device=self.device)
        target = torch.zeros(b, POLICY_SIZE, device=self.device)
        target.scatter_add_(1, self.pol[i].long(), self.prob[i].float())  # padding has prob 0
        return self.planes[i].float(), target, self.value[i]


def train_selfplay_gpu(
    config: Optional[Config] = None,
    iterations: int = 40,
    games_per_iter: int = 256,
    slots: int = 256,
    sims: int = 400,
    train_steps: int = 200,
    batch_size: int = 256,
    lr: float = 1e-4,
    buffer_capacity: int = 300_000,
    init_checkpoint: Optional[str] = "models/supervised.pt",
    out_name: str = "selfplay.pt",
    best_name: str = "best.pt",
    temperature_moves: int = 18,
    sup_fraction: float = 0.5,
    anchor_size: int = 1_000_000,
    resign: bool = True,
    resign_threshold: float = 0.95,
    resign_streak: int = 3,
    complete_fraction: float = 0.08,
    arena_every: int = 3,
    arena_games: int = 400,
    arena_sims: Optional[int] = None,
    arena_book_fraction: float = 0.7,
    arena_opening_min: int = 6,
    arena_opening_max: int = 24,
    arena_temp_moves: int = 0,
    gate_threshold: float = 0.55,
    gate_min_games: int = 300,
    gate_require_significance: bool = True,
    search_amp: bool = True,
    start_iter: int = 0,
    progress: Optional[ProgressLogger] = None,
) -> str:
    config = config or Config()
    config.ensure_dirs()
    progress = progress or ProgressLogger(
        os.path.join(config.logs_dir, "selfplay.jsonl"), append=start_iter > 0
    )
    device = config.device
    arena_sims = arena_sims or sims

    if init_checkpoint and os.path.exists(init_checkpoint):
        champion, _ = load_checkpoint(init_checkpoint, device=device)
    else:
        champion = build_model(config.model, device=device)
    candidate = build_model(champion.config, device=device)
    candidate.load_state_dict(champion.state_dict())
    optimizer = torch.optim.Adam(candidate.parameters(), lr=lr, weight_decay=1e-4)
    buffer = GpuReplayBuffer(buffer_capacity, device)

    anchor = None
    if sup_fraction > 0.0:
        ds = StockfishDataset(config.data_dir)
        if len(ds) > 0:
            anchor = GpuAnchor(ds, anchor_size, device)

    out_path = os.path.join(config.models_dir, out_name)
    best_path = os.path.join(config.models_dir, best_name)
    book = try_load_opening_book(config.opening_book_path)
    pairs = max(1, arena_games // 2)
    champ_version = 0

    progress.log({
        "event": "start", "mode": "selfplay", "engine": "gpu", "iterations": iterations,
        "start_iter": start_iter, "games_per_iter": games_per_iter, "sims": sims, "device": device,
        "lr": lr, "sup_fraction": sup_fraction if anchor else 0.0,
        "anchor_samples": anchor.n if anchor else 0, "arena_every": arena_every,
        "arena_games": arena_games, "gate_threshold": gate_threshold,
        "gate_min_games": gate_min_games, "temperature_moves": temperature_moves,
        "resign": resign, "search_amp": search_amp,
    })
    save_checkpoint(best_path, champion, meta={"iter": start_iter - 1, "note": "init"})

    for it in range(start_iter, iterations):
        torch.manual_seed(1000 + it)
        t0 = time.time()

        # ---- 1. self-play from the CHAMPION (all games concurrent on the GPU) -------------------
        def log_game(ev: dict, it=it) -> None:
            progress.log({"event": "selfplay_game", "iter": it, "games_per_iter": games_per_iter, **ev})

        stats = gpu_selfplay(
            champion, buffer, games_per_iter, slots=slots, sims=sims,
            temperature_moves=temperature_moves, resign=resign, resign_threshold=resign_threshold,
            resign_streak=resign_streak, complete_fraction=complete_fraction,
            amp=search_amp, on_game=log_game,
            on_tick=lambda it=it: progress.log({"event": "heartbeat", "iter": it, "phase": "selfplay"}),
        )
        progress.log({"event": "selfplay_done", "iter": it, "seconds": round(time.time() - t0),
                      "buffer": len(buffer), **stats})

        # ---- 2. train the CANDIDATE (self-play + supervised anchor) ------------------------------
        candidate.train()
        if len(buffer) >= batch_size:
            n_sup = int(round(batch_size * sup_fraction)) if anchor else 0
            n_buf = batch_size - n_sup
            for step in range(train_steps):
                bp, bpi, bv = buffer.sample_batch(n_buf)
                if n_sup:
                    sp_, spi, sv = anchor.sample_batch(n_sup)
                    planes = torch.cat([bp, sp_])
                    target_policy = torch.cat([bpi, spi])
                    target_value = torch.cat([bv, sv])
                else:
                    planes, target_policy, target_value = bp, bpi, bv
                logits, value = candidate(planes)
                p_loss = policy_loss(logits, target_policy)
                v_loss = F.mse_loss(value, target_value)
                optimizer.zero_grad()
                (p_loss + v_loss).backward()
                optimizer.step()
                if step % 25 == 0:
                    progress.log({"event": "train_step", "iter": it, "step": step,
                                  "policy_loss": round(p_loss.item(), 4),
                                  "value_loss": round(v_loss.item(), 4)})
        save_checkpoint(out_path, candidate, meta={"iter": it})

        # ---- 3. arena gating ------------------------------------------------------------------------
        if arena_every and (it + 1) % arena_every == 0 and len(buffer) >= batch_size:
            t1 = time.time()
            rng = random.Random(20240703 + it)
            openings = build_mixed_opening_pool(
                rng, size=pairs, book=book, book_fraction=arena_book_fraction,
                min_plies=arena_opening_min, max_plies=arena_opening_max,
                random_min_plies=4, random_max_plies=12,
            )  # one fresh opening per pair so the games are independent
            score, w, d, l = gpu_arena(
                candidate, champion, openings, sims=arena_sims, temperature_moves=arena_temp_moves,
                amp=search_amp,
                on_tick=lambda it=it: progress.log({"event": "heartbeat", "iter": it, "phase": "arena"}),
                on_game=lambda ev, it=it: progress.log({"event": "arena_game", "iter": it, **ev}),
            )
            promote, reason = _gate_promote(w, d, l, gate_threshold, gate_min_games,
                                            gate_require_significance)
            common = {"iter": it, "arena_score": round(score, 3), "gate_reason": reason,
                      "wins": w, "draws": d, "losses": l, "arena_openings": len(openings),
                      "arena_seconds": round(time.time() - t1)}
            if promote:
                champion.load_state_dict(candidate.state_dict())
                champ_version += 1
                save_checkpoint(best_path, champion, meta={"iter": it, "arena_score": score})
                progress.log({"event": "promote", "champ_version": champ_version, **common})
            else:
                candidate.load_state_dict(champion.state_dict())  # reject: revert the candidate
                progress.log({"event": "arena_reject", **common})

        progress.log({"event": "iter_done", "iter": it, "seconds": round(time.time() - t0)})

    progress.log({"event": "done", "mode": "selfplay", "checkpoint": best_path,
                  "champ_version": champ_version})
    return best_path
