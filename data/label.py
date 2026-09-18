"""Label positions with Stockfish and write resumable .npz shards.

Each shard stores planes, values and a flattened sparse policy (indices+probs
with per-sample lengths). Labeling can run across multiple worker processes
(one Stockfish per worker) to exploit all CPU cores, and reports live progress
with elapsed time, throughput and ETA so you can see it is actually working.

Resumability: shards accumulate, and on restart with the *same* generated FEN
set (same ``--generate N --seed S``) already-labeled positions are skipped, so a
long run that is interrupted can be continued.

Teachers: Stockfish (CPU, depth-limited) or lc0 (GPU, node-limited). With
``chain_len > 1`` each start position is followed by the teacher's own sampled
moves, yielding up to ``chain_len`` consecutive labeled positions per task. That
gives realistic game positions (not just random-walk ones) at no extra search
cost, since the search that labels a position also picks the move that leaves it.
"""
from __future__ import annotations

import glob
import os
import random
import time
from multiprocessing import Pool
from typing import Iterable, List, Optional, Tuple

import chess
import numpy as np

from engine.config import Config
from engine.encoding import board_to_planes, move_to_index
from teacher import make_teacher

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


# A chain stops early once the teacher sees a decided position (|value| above this).
DECIDED_VALUE = 0.98

Sample = Tuple[np.ndarray, float, List[int], List[float]]


def _init_worker(config, teacher_kind, chain_len, teacher_opts):
    _WORKER["teacher"] = make_teacher(teacher_kind, config, **teacher_opts)
    _WORKER["chain_len"] = chain_len
    _WORKER["rng"] = random.Random(os.getpid() ^ int(time.time() * 1000))


def _label_task(fen: str) -> List[Sample]:
    """Label ``fen`` and, in chain mode, the positions that follow it."""
    teacher = _WORKER["teacher"]
    chain_len: int = _WORKER["chain_len"]
    rng: random.Random = _WORKER["rng"]
    board = chess.Board(fen)
    out: List[Sample] = []
    for step in range(chain_len):
        if board.is_game_over():
            break
        label = teacher.label(board)
        if not label["policy_indices"]:
            break
        value = float(label["value"])
        out.append((
            board_to_planes(board), value,
            list(label["policy_indices"]), list(label["policy_probs"]),
        ))
        if step == chain_len - 1 or abs(value) > DECIDED_VALUE:
            break
        by_index = {move_to_index(m): m for m in board.legal_moves}
        choices = [(i, p) for i, p in zip(label["policy_indices"], label["policy_probs"]) if i in by_index]
        if not choices:
            break
        idx = rng.choices([i for i, _ in choices], weights=[p for _, p in choices])[0]
        board.push(by_index[idx])
    return out


class _TargetReached(Exception):
    """Raised inside ``collect`` to stop labeling once ``target_samples`` exist."""


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
    teacher: str = "stockfish",
    nodes: int = 400,
    minibatch: int = 32,
    chain_len: int = 1,
    resume: bool = True,
    target_samples: Optional[int] = None,
    workers: int = 1,
    heartbeat_seconds: float = 5.0,
    progress=None,
) -> List[str]:
    """Label ``fens`` and write shards. Returns the list of shard paths written.

    ``resume`` skips the first ``already_labeled // chain_len`` start positions, which
    is only meaningful when re-running the *same* position list; pass ``resume=False``
    to append shards for a fresh list (e.g. a new seed). ``target_samples`` stops once
    the directory holds that many samples in total (existing shards included).

    ``workers`` > 1 runs that many teacher processes in parallel. ``fens`` are
    start positions; with ``chain_len`` > 1 each yields up to that many samples.
    ``progress`` is an optional callable(dict) for streaming status updates.
    """
    config = config or Config()
    config.ensure_dirs()
    all_fens = list(fens)
    chain_len = max(1, chain_len)
    already = count_labeled(config.data_dir)
    # Upper bound on samples (chains can end early).
    total = target_samples or len(all_fens) * chain_len
    # Resume: skip start positions already covered by existing shards.
    todo = all_fens[already // chain_len:] if resume else all_fens
    if target_samples and already >= target_samples:
        todo = []
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

    def collect(results: List[Sample]) -> None:
        nonlocal done, last_beat
        for planes, value, idxs, probs in results:
            buf_planes.append(planes)
            buf_values.append(value)
            buf_idx.extend(idxs)
            buf_prob.extend(probs)
            buf_len.append(len(idxs))
            done += 1
            if len(buf_planes) >= shard_size:
                flush()
        if target_samples and already + done >= target_samples:
            raise _TargetReached

        now = time.time()
        if progress and (now - last_beat) >= heartbeat_seconds:
            elapsed = now - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            budget = len(todo) * chain_len
            if target_samples:
                budget = min(budget, target_samples - already)
            remaining = max(budget - done, 0)
            eta = remaining / rate if rate > 0 else 0.0
            progress({
                "event": "label",
                "done": already + done,
                "total": total,
                "session_done": done,
                "session_total": min(len(todo) * chain_len, (target_samples - already) if target_samples else 1 << 60),
                "rate_pos_per_sec": round(rate, 2),
                "elapsed": _fmt_duration(elapsed),
                "eta": _fmt_duration(eta),
                "shards_written": len(written),
            })
            last_beat = now

    # Preflight: make sure Stockfish actually launches before spawning a pool of
    # workers (a failing worker initializer would otherwise hang silently).
    teacher_opts = dict(
        depth=depth, movetime=movetime, multipv=multipv, threads=threads,
        hash_mb=hash_mb, nodes=nodes, minibatch=minibatch,
    )
    try:
        probe = make_teacher(teacher, config, **teacher_opts)
        probe.close()
    except Exception as exc:  # noqa: BLE001
        if progress:
            exe = config.lc0_path if teacher == "lc0" else config.stockfish_path
            progress({
                "event": "error", "stage": teacher,
                "message": f"Could not start {teacher} ('{exe}'): {exc}",
                "hint": "Set LC0_PATH/LC0_WEIGHTS or STOCKFISH_PATH and restart the server.",
            })
        raise

    if progress:
        progress({
            "event": "start", "mode": "label", "total": total,
            "already_labeled": already, "to_label": len(todo),
            "workers": workers, "teacher": teacher, "chain_len": chain_len,
            "depth": depth, "nodes": nodes, "multipv": multipv,
        })

    if not todo:
        if progress:
            progress({"event": "label_done", "done": already, "total": total,
                      "shards": len(written), "note": "already complete"})
        return written

    init_args = (config, teacher, chain_len, teacher_opts)

    if workers and workers > 1:
        with Pool(processes=workers, initializer=_init_worker, initargs=init_args) as pool:
            try:
                for result in pool.imap_unordered(_label_task, todo, chunksize=2):
                    collect(result)
            except _TargetReached:
                pass  # leaving the with-block terminates the workers
    else:
        _init_worker(*init_args)
        try:
            for fen in todo:
                collect(_label_task(fen))
        except _TargetReached:
            pass
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
