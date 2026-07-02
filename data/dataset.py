"""Datasets and the self-play replay buffer."""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.label import _planes_path, existing_shards
from engine.encoding import POLICY_SIZE


def _dense_from_sparse(indices: np.ndarray, probs: np.ndarray) -> np.ndarray:
    vec = np.zeros(POLICY_SIZE, dtype=np.float32)
    if len(indices):
        vec[indices] = probs
    return vec


class _Shard:
    """One shard: planes memory-mapped from disk; value/policy metadata in RAM."""

    def __init__(self, planes_path: str, meta_path: str):
        # mmap_mode='r' keeps planes on disk and pages them in on demand.
        self.planes = np.load(planes_path, mmap_mode="r")
        with np.load(meta_path) as meta:
            self.values = meta["values"]
            self.pol_idx = meta["pol_idx"]
            self.pol_prob = meta["pol_prob"]
            pol_len = meta["pol_len"]
        # Precompute the start offset of each sample's sparse policy.
        self.offsets = np.zeros(len(pol_len) + 1, dtype=np.int64)
        np.cumsum(pol_len, out=self.offsets[1:])
        self.n = int(self.planes.shape[0])

    def sample(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        planes = np.asarray(self.planes[i], dtype=np.float32)
        return planes, self.pol_idx[start:end], self.pol_prob[start:end], float(self.values[i])


class StockfishDataset(Dataset):
    """Lazy, memory-mapped dataset over Stockfish-labeled shards.

    Only a small per-shard metadata footprint (values + sparse policy) is held in
    RAM; the large planes tensors are memory-mapped and paged in on access, so the
    corpus can be far larger than available memory.
    """

    def __init__(self, data_dir: str, limit: Optional[int] = None):
        self.shards: List[_Shard] = []
        # Global sample index -> (shard_id, local_index).
        self.index: List[Tuple[int, int]] = []

        for meta_path in existing_shards(data_dir):
            base = os.path.basename(meta_path).replace(".meta.npz", "")
            shard_idx = int(base.replace("labels_", ""))
            planes_path = _planes_path(data_dir, shard_idx)
            if not os.path.exists(planes_path):
                continue
            shard = _Shard(planes_path, meta_path)
            sid = len(self.shards)
            self.shards.append(shard)
            for li in range(shard.n):
                self.index.append((sid, li))
                if limit and len(self.index) >= limit:
                    break
            if limit and len(self.index) >= limit:
                break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sid, li = self.index[idx]
        planes, pol_idx, pol_prob, value = self.shards[sid].sample(li)
        policy = _dense_from_sparse(pol_idx, pol_prob)
        return (
            torch.from_numpy(planes),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32),
        )


class ReplayBuffer:
    """In-memory replay buffer for self-play samples (planes, dense policy, z)."""

    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, planes: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.buffer.append((planes.astype(np.float32), policy.astype(np.float32), float(value)))

    def add_game(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        for s in samples:
            self.add(*s)

    def sample_batch(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        planes = torch.from_numpy(np.stack([b[0] for b in batch]))
        policy = torch.from_numpy(np.stack([b[1] for b in batch]))
        value = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return planes, policy, value
