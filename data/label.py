"""Label positions with Stockfish and write resumable .npz shards.

Each shard stores planes, values and a flattened sparse policy (indices+probs
with per-sample lengths). Labeling can run across multiple worker processes
(one Stockfish per worker) to exploit all CPU cores, and reports live progress
with elapsed time, throughput and ETA so you can see it is actually working.

Resumability: shards accumulate, and on restart with the *same* generated FEN
set (same ``--generate N --seed S``) already-labeled positions are skipped, so a
long run that is interrupted can be continued.
"""
from __future__ import annotations

import glob
import os
import time
from multiprocessing import Pool
from typing import Iterable, List, Optional, Tuple

import chess
import numpy as np

from engine.config import Config
from engine.encoding import board_to_planes
from teacher.stockfish import StockfishTeacher

SHARD_PREFIX = "labels_"

# Per-worker Stockfish instance (set by the pool initializer).
_WORKER: dict = {}


def _planes_path(data_dir: str, index: int) -> str:
    return os.path.join(data_dir, f"{SHARD_PREFIX}{index:05d}.planes.npy")


def _meta_path(data_dir: str, index: int) -> str:
    return os.path.join(data_dir, f"{SHARD_PREFIX}{index:05d}.meta.npz")


def existing_shards(data_dir: str) -> List[str]:
    """Return sorted meta-shard paths (one per shard)."""
    return sorted(glob.glob(os.path.join(data_dir, f"{SHARD_PREFIX}*.meta.npz")))


def count_labeled(data_dir: str) -> int:
    """Total number of already-labeled positions across existing shards."""
    total = 0
    for shard in existing_shards(data_dir):
        try:
            with np.load(shard) as data:
                total += int(data["values"].shape[0])
        except Exception:
            pass
    return total


def _init_worker(path, depth, movetime, multipv, threads, hash_mb):
    _WORKER["teacher"] = StockfishTeacher(
        path, depth=depth, movetime=movetime, multipv=multipv,
        threads=threads, hash_mb=hash_mb,
    )


def _label_one(fen: str) -> Tuple[np.ndarray, float, List[int], List[float]]:
    teacher: StockfishTeacher = _WORKER["teacher"]
    board = chess.Board(fen)
    label = teacher.label(board)
    return (
        board_to_planes(board),
        float(label["value"]),
        list(label["policy_indices"]),
        list(label["policy_probs"]),
    )


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def label_positions(
    fens: Iterable[str],
    config: Optional[Config] = None,
    shard_size: int = 2000,
    depth: int = 12,
    movetime: Optional[float] = None,
    multipv: int = 4,
    threads: int = 1,
    hash_mb: int = 128,
    workers: int = 1,
    heartbeat_seconds: float = 5.0,
    progress=None,
) -> List[str]:
    """Label ``fens`` and write shards. Returns the list of shard paths written.

    ``workers`` > 1 runs that many Stockfish worker processes in parallel.
    ``progress`` is an optional callable(dict) for streaming status updates.
    """
    config = config or Config()
    config.ensure_dirs()
    all_fens = list(fens)
    total = len(all_fens)

    # Resume: skip positions already covered by existing shards.
    already = count_labeled(config.data_dir)
    todo = all_fens[already:]
    start_shard = len(existing_shards(config.data_dir))
    written: List[str] = []

    buf_planes: List[np.ndarray] = []
    buf_values: List[float] = []
    buf_idx: List[int] = []
    buf_prob: List[float] = []
    buf_len: List[int] = []
    state = {"shard_index": start_shard}

    def flush() -> None:
        if not buf_planes:
            return
        idx = state["shard_index"]
        # Planes are all 0/1: store as uint8 in an uncompressed, memory-mappable
        # .npy so the dataset can lazily read positions without loading the whole
        # corpus into RAM. Small value/policy metadata goes in a compressed sidecar.
        np.save(
            _planes_path(config.data_dir, idx),
            np.stack(buf_planes).astype(np.uint8),
        )
        meta_path = _meta_path(config.data_dir, idx)
        np.savez_compressed(
            meta_path,
            values=np.asarray(buf_values, dtype=np.float32),
            pol_idx=np.asarray(buf_idx, dtype=np.int32),
            pol_prob=np.asarray(buf_prob, dtype=np.float32),
            pol_len=np.asarray(buf_len, dtype=np.int32),
        )
        written.append(meta_path)
        state["shard_index"] += 1
        buf_planes.clear()
        buf_values.clear()
        buf_idx.clear()
        buf_prob.clear()
        buf_len.clear()
        if progress:
            progress({"event": "shard", "path": os.path.basename(meta_path),
                      "shards_written": len(written)})

    start_time = time.time()
    last_beat = start_time
    done = 0

    def collect(result) -> None:
        nonlocal done, last_beat
        planes, value, idxs, probs = result
        buf_planes.append(planes)
        buf_values.append(value)
        buf_idx.extend(idxs)
        buf_prob.extend(probs)
        buf_len.append(len(idxs))
        done += 1

        if len(buf_planes) >= shard_size:
            flush()

        now = time.time()
        if progress and (now - last_beat) >= heartbeat_seconds:
            elapsed = now - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = len(todo) - done
            eta = remaining / rate if rate > 0 else 0.0
            progress({
                "event": "label",
                "done": already + done,
                "total": total,
                "session_done": done,
                "session_total": len(todo),
                "rate_pos_per_sec": round(rate, 2),
                "elapsed": _fmt_duration(elapsed),
                "eta": _fmt_duration(eta),
                "shards_written": len(written),
            })
            last_beat = now

    # Preflight: make sure Stockfish actually launches before spawning a pool of
    # workers (a failing worker initializer would otherwise hang silently).
    try:
        probe = StockfishTeacher(
            config.stockfish_path, depth=depth, movetime=movetime,
            multipv=multipv, threads=threads, hash_mb=hash_mb,
        )
        probe.close()
    except Exception as exc:  # noqa: BLE001
        if progress:
            progress({
                "event": "error", "stage": "stockfish",
                "message": f"Could not start Stockfish at '{config.stockfish_path}': {exc}",
                "hint": "Set STOCKFISH_PATH to your Stockfish binary and restart the server.",
            })
        raise

    if progress:
        progress({
            "event": "start", "mode": "label", "total": total,
            "already_labeled": already, "to_label": len(todo),
            "workers": workers, "depth": depth, "multipv": multipv,
        })

    if not todo:
        if progress:
            progress({"event": "label_done", "done": already, "total": total,
                      "shards": len(written), "note": "already complete"})
        return written

    init_args = (config.stockfish_path, depth, movetime, multipv, threads, hash_mb)

    if workers and workers > 1:
        with Pool(processes=workers, initializer=_init_worker, initargs=init_args) as pool:
            for result in pool.imap_unordered(_label_one, todo, chunksize=4):
                collect(result)
    else:
        _init_worker(*init_args)
        try:
            for fen in todo:
                collect(_label_one(fen))
        finally:
            _WORKER["teacher"].close()

    flush()

    if progress:
        elapsed = time.time() - start_time
        progress({
            "event": "label_done", "done": already + done, "total": total,
            "shards": len(written),
            "elapsed": _fmt_duration(elapsed),
            "avg_rate_pos_per_sec": round(done / elapsed, 2) if elapsed > 0 else 0.0,
        })
    return written
