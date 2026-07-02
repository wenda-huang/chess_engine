"""Phase 1: supervised bootstrap training on Stockfish-labeled positions."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.dataset import StockfishDataset
from engine.config import Config
from engine.model import build_model, load_checkpoint, save_checkpoint
from train.progress import ProgressLogger


def policy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted distribution and target distribution."""
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def train_supervised(
    config: Optional[Config] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.05,
    limit: Optional[int] = None,
    resume: Optional[str] = None,
    out_name: str = "supervised.pt",
    progress: Optional[ProgressLogger] = None,
) -> str:
    config = config or Config()
    config.ensure_dirs()
    progress = progress or ProgressLogger(os.path.join(config.logs_dir, "supervised.jsonl"))
    device = config.device

    dataset = StockfishDataset(config.data_dir, limit=limit)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No labeled data found in '{config.data_dir}'. Run the labeling step first."
        )

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    if resume and os.path.exists(resume):
        model, _ = load_checkpoint(resume, device=device)
    else:
        model = build_model(config.model, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    progress.log(
        {
            "event": "start",
            "mode": "supervised",
            "samples": len(dataset),
            "train": n_train,
            "val": n_val,
            "epochs": epochs,
            "device": device,
            "blocks": model.config.num_blocks,
            "channels": model.config.channels,
            "params_millions": round(n_params / 1e6, 2),
        }
    )

    out_path = os.path.join(config.models_dir, out_name)
    best_val = float("inf")
    step = 0

    for epoch in range(epochs):
        model.train()
        running_p, running_v, count = 0.0, 0.0, 0
        for planes, target_policy, target_value in train_loader:
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

            running_p += p_loss.item() * planes.size(0)
            running_v += v_loss.item() * planes.size(0)
            count += planes.size(0)
            step += 1
            if step % 20 == 0:
                progress.log(
                    {
                        "event": "step",
                        "epoch": epoch,
                        "step": step,
                        "policy_loss": round(p_loss.item(), 4),
                        "value_loss": round(v_loss.item(), 4),
                    }
                )

        val_p, val_v = _validate(model, val_loader, device)
        progress.log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_policy_loss": round(running_p / max(count, 1), 4),
                "train_value_loss": round(running_v / max(count, 1), 4),
                "val_policy_loss": round(val_p, 4),
                "val_value_loss": round(val_v, 4),
            }
        )

        val_total = val_p + val_v
        if val_total < best_val:
            best_val = val_total
            save_checkpoint(out_path, model, meta={"epoch": epoch, "val_loss": val_total})

    # Always save the final model too (in case val split was tiny).
    save_checkpoint(out_path, model, meta={"epoch": epochs - 1, "final": True})
    progress.log({"event": "done", "mode": "supervised", "checkpoint": out_path})
    return out_path


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    tot_p, tot_v, count = 0.0, 0.0, 0
    for planes, target_policy, target_value in loader:
        planes = planes.to(device)
        target_policy = target_policy.to(device)
        target_value = target_value.to(device)
        logits, value = model(planes)
        tot_p += policy_loss(logits, target_policy).item() * planes.size(0)
        tot_v += F.mse_loss(value, target_value).item() * planes.size(0)
        count += planes.size(0)
    return tot_p / max(count, 1), tot_v / max(count, 1)
