"""Phase 1: supervised bootstrap training on teacher-labeled positions (Stockfish or lc0)."""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import StockfishDataset
from engine.config import Config
from engine.encoding import POLICY_SIZE
from engine.model import build_model, load_checkpoint, save_checkpoint
from train.progress import ProgressLogger


def policy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted distribution and target distribution."""
    logp = F.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


class _Prefetcher:
    """Assemble batches on a background thread so the GPU never waits on numpy.

    numpy releases the GIL for the big gathers, so one thread keeps up easily.
    """

    def __init__(self, make_batches, depth: int = 6):
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._run, args=(make_batches,), daemon=True)
        self._thread.start()

    def _run(self, make_batches) -> None:
        try:
            for item in make_batches():
                self._queue.put(item)
            self._queue.put(None)
        except BaseException as exc:  # noqa: BLE001 - surface in the consumer
            self._queue.put(exc)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def _batches(dataset: StockfishDataset, order: np.ndarray, batch_size: int):
    for start in range(0, len(order), batch_size):
        yield dataset.batch(order[start:start + batch_size])


def _to_device(batch, device: str):
    """Move a sparse batch to ``device`` and densify the policy target there."""
    planes, pol_idx, pol_prob, values = batch
    planes = torch.from_numpy(planes).to(device, non_blocking=True).float()
    idx = torch.from_numpy(pol_idx).to(device, non_blocking=True)
    prob = torch.from_numpy(pol_prob).to(device, non_blocking=True)
    target = torch.zeros(idx.shape[0], POLICY_SIZE, device=device)
    target.scatter_add_(1, idx, prob)  # padding has prob 0, so index 0 is unaffected
    values = torch.from_numpy(values).to(device, non_blocking=True)
    return planes, target, values


def _lr_at(step: int, total: int, base_lr: float, warmup: int, cosine: bool, floor: float = 0.02) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if not cosine:
        return base_lr
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


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
    num_workers: int = 0,  # unused: batches are assembled on a background thread
    progress: Optional[ProgressLogger] = None,
    amp: bool = True,
    cosine: bool = True,
    warmup_steps: int = 300,
    ckpt_every: int = 1,
    patience: int = 3,
) -> str:
    """Supervised training on labeled shards.

    ``amp`` runs the forward pass in bfloat16 on CUDA/ROCm (losses stay fp32);
    ``cosine`` decays the learning rate to 2% of ``lr`` over the whole run after a
    short linear warmup, which matters a lot for the final loss on a fixed dataset.
    """
    config = config or Config()
    config.ensure_dirs()
    progress = progress or ProgressLogger(os.path.join(config.logs_dir, "supervised.jsonl"))
    device = config.device
    use_amp = amp and device == "cuda"

    dataset = StockfishDataset(config.data_dir, limit=limit)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No labeled data found in '{config.data_dir}'. Run the labeling step first."
        )

    n_val = max(1, int(len(dataset) * val_fraction))
    perm = np.random.RandomState(0).permutation(len(dataset))
    val_idx, train_idx = np.sort(perm[:n_val]), perm[n_val:]
    n_train = len(train_idx)

    if resume and os.path.exists(resume):
        model, resume_meta = load_checkpoint(resume, device=device)
        epoch_offset = int((resume_meta or {}).get("epoch", -1)) + 1  # cumulative epoch count
    else:
        model = build_model(config.model, device=device)
        epoch_offset = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = epochs * steps_per_epoch

    n_params = sum(p.numel() for p in model.parameters())
    progress.log(
        {
            "event": "start",
            "mode": "supervised",
            "samples": len(dataset),
            "train": n_train,
            "val": n_val,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "amp_bf16": use_amp,
            "cosine_lr": cosine,
            "blocks": model.config.num_blocks,
            "channels": model.config.channels,
            "params_millions": round(n_params / 1e6, 2),
        }
    )

    out_path = os.path.join(config.models_dir, out_name)
    best_val = float("inf")
    bad_epochs = 0
    last_epoch = 0
    step = 0
    rng = np.random.RandomState(1)
    t_start = time.time()

    def forward(planes):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, value = model(planes)
        return logits.float(), value.float()

    for epoch in range(epochs):
        model.train()
        running_p, running_v, count = 0.0, 0.0, 0
        order = rng.permutation(train_idx)
        t_epoch = time.time()
        for batch in _Prefetcher(lambda: _batches(dataset, order, batch_size)):
            planes, target_policy, target_value = _to_device(batch, device)

            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, total_steps, lr, warmup_steps, cosine)

            logits, value = forward(planes)
            p_loss = policy_loss(logits, target_policy)
            v_loss = F.mse_loss(value, target_value)
            loss = p_loss + v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = planes.size(0)
            step += 1
            if step % 20 == 0:
                # .item() syncs with the GPU, so only do it on logged steps.
                pl, vl = p_loss.item(), v_loss.item()
                running_p += pl * n
                running_v += vl * n
                count += n
                progress.log(
                    {
                        "event": "step",
                        "epoch": epoch,
                        "step": step,
                        "of": total_steps,
                        "policy_loss": round(pl, 4),
                        "value_loss": round(vl, 4),
                        "lr": round(optimizer.param_groups[0]["lr"], 6),
                        "samples_per_sec": round(step * batch_size / (time.time() - t_start)),
                    }
                )

        val_p, val_v = _validate(model, dataset, val_idx, batch_size, device, forward)
        progress.log(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_policy_loss": round(running_p / max(count, 1), 4),
                "train_value_loss": round(running_v / max(count, 1), 4),
                "val_policy_loss": round(val_p, 4),
                "val_value_loss": round(val_v, 4),
                "epoch_seconds": round(time.time() - t_epoch),
            }
        )

        val_total = val_p + val_v
        last_epoch = epoch
        improved = val_total < best_val
        if improved:
            best_val = val_total
            bad_epochs = 0
            save_checkpoint(out_path, model, meta={"epoch": epoch_offset + epoch, "val_loss": val_total})

        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            # Crash-recovery snapshot (weights only): resume with --resume <name>_ckpt.pt.
            save_checkpoint(os.path.splitext(out_path)[0] + "_ckpt.pt", model,
                            meta={"epoch": epoch_offset + epoch, "val_loss": val_total})

        if not improved:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                progress.log({"event": "early_stop", "epoch": epoch, "best_val_loss": round(best_val, 4)})
                break

    # Keep the last-epoch weights separately; ``out_path`` holds the best-validation epoch.
    final_path = os.path.splitext(out_path)[0] + "_final.pt"
    save_checkpoint(final_path, model, meta={"epoch": epoch_offset + last_epoch, "final": True})
    progress.log({"event": "done", "mode": "supervised", "checkpoint": out_path,
                  "final_checkpoint": final_path, "best_val_loss": round(best_val, 4)})
    return out_path


@torch.no_grad()
def _validate(model, dataset, val_idx, batch_size, device, forward):
    model.eval()
    tot_p, tot_v, count = 0.0, 0.0, 0
    for batch in _Prefetcher(lambda: _batches(dataset, val_idx, batch_size)):
        planes, target_policy, target_value = _to_device(batch, device)
        logits, value = forward(planes)
        n = planes.size(0)
        tot_p += policy_loss(logits, target_policy).item() * n
        tot_v += F.mse_loss(value, target_value).item() * n
        count += n
    return tot_p / max(count, 1), tot_v / max(count, 1)
