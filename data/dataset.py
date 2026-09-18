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


def _raise_fd_limit() -> None:
    """Best-effort raise of the open-file limit.

    Each shard is held open as a memory-mapped file, so a large corpus (hundreds
    of shards) can bump into the default ``ulimit -n`` of 1024. That starves
    later ``open`` calls -- including CUDA opening its device nodes, which then
    surfaces as a misleading ``cudaErrorDevicesUnavailable``. Raising the limit
    up front avoids it. No-op on non-POSIX platforms or if not permitted.
    """
    try:
        import resource
    except ImportError:
        return  # Windows
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 1_048_576
        if hard != resource.RLIM_INFINITY and hard < want:
            # Try to lift the hard cap too (permitted as root); fall back if not.
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, want))
                return
            except (ValueError, OSError):
                pass
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


class StockfishDataset(Dataset):
    """Lazy, memory-mapped dataset over labeled shards (Stockfish or lc0 teacher).

    Only per-sample metadata (values + sparse policy, ~100 bytes/sample) is held in
    RAM as flat numpy arrays; the large planes tensors stay memory-mapped and are
    paged in on access, so the corpus can be far larger than available memory.

    ``ds[i]`` returns one dense ``(planes, policy, value)`` sample. ``batch(idx)``
    is the fast path for training: it gathers many samples with vectorized numpy and
    returns the policy sparse (padded), so no per-sample dense vector is built.
    """

    def __init__(self, data_dir: str, limit: Optional[int] = None):
        _raise_fd_limit()
        self.planes: List[np.ndarray] = []  # one memmap per shard
        counts: List[int] = []
        values, pol_len, pol_idx, pol_prob = [], [], [], []
        total = 0

        for meta_path in existing_shards(data_dir):
            if limit and total >= limit:
                break
            base = os.path.basename(meta_path).replace(".meta.npz", "")
            planes_path = _planes_path(data_dir, int(base.replace("labels_", "")))
            if not os.path.exists(planes_path):
                continue
            planes = np.load(planes_path, mmap_mode="r")
            with np.load(meta_path) as meta:
                v, ln = meta["values"], meta["pol_len"]
                pi, pp = meta["pol_idx"], meta["pol_prob"]
            n = int(planes.shape[0])
            if limit and total + n > limit:  # keep the sparse policy in step with the cut
                n = limit - total
                v, ln = v[:n], ln[:n]
                pi, pp = pi[: int(ln.sum())], pp[: int(ln.sum())]
            self.planes.append(planes)
            counts.append(n)
            values.append(v)
            pol_len.append(ln)
            pol_idx.append(pi)
            pol_prob.append(pp)
            total += n

        self.n = total
        self.shard_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.values = np.concatenate(values).astype(np.float32) if values else np.zeros(0, np.float32)
        lens = np.concatenate(pol_len).astype(np.int64) if pol_len else np.zeros(0, np.int64)
        self.pol_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.pol_idx = np.concatenate(pol_idx).astype(np.int64) if pol_idx else np.zeros(0, np.int64)
        self.pol_prob = np.concatenate(pol_prob).astype(np.float32) if pol_prob else np.zeros(0, np.float32)

    def __len__(self) -> int:
        return self.n

    def _planes_for(self, idx: np.ndarray) -> np.ndarray:
        """uint8 planes for ``idx`` (any order), gathered shard by shard."""
        out = np.empty((len(idx),) + self.planes[0].shape[1:], dtype=np.uint8)
        sid = np.searchsorted(self.shard_start, idx, side="right") - 1
        for s in np.unique(sid):
            rows = np.nonzero(sid == s)[0]
            local = idx[rows] - self.shard_start[s]
            order = np.argsort(local)  # sorted reads are friendlier to the mmap
            out[rows[order]] = self.planes[s][local[order]]
        return out

    def batch(self, idx: np.ndarray):
        """Return ``(planes uint8 [B,18,8,8], pol_idx int64 [B,K], pol_prob f32 [B,K], values f32 [B])``.

        Policy rows are zero-padded to the longest in the batch (padding has prob 0).
        """
        idx = np.asarray(idx, dtype=np.int64)
        starts = self.pol_off[idx]
        lens = self.pol_off[idx + 1] - starts
        k = max(int(lens.max()), 1)
        cols = np.arange(k)[None, :]
        mask = cols < lens[:, None]
        pos = np.minimum(starts[:, None] + cols, max(len(self.pol_idx) - 1, 0))
        pol_idx = np.where(mask, self.pol_idx[pos], 0)
        pol_prob = np.where(mask, self.pol_prob[pos], 0.0).astype(np.float32)
        return self._planes_for(idx), pol_idx, pol_prob, self.values[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planes, pol_idx, pol_prob, values = self.batch(np.array([idx]))
        n = int(self.pol_off[idx + 1] - self.pol_off[idx])
        policy = _dense_from_sparse(pol_idx[0, :n], pol_prob[0, :n])
        return (
            torch.from_numpy(planes[0].astype(np.float32)),
            torch.from_numpy(policy),
            torch.tensor(float(values[0]), dtype=torch.float32),
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
