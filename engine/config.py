"""Global configuration and device auto-detection.

The same config object is used everywhere (training, self-play, serving) so
the exact same code runs on a CPU laptop and on a cloud GPU box -- only the
values here (or the corresponding environment variables) change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


def get_device(preferred: Optional[str] = None) -> str:
    """Return the best available torch device string.

    Order of preference: explicit arg / env -> cuda -> mps (Apple) -> cpu.
    Imports torch lazily so that modules that only need paths do not pay the
    (slow) torch import cost.
    """
    if preferred:
        return preferred
    env = os.environ.get("CHESSAI_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class ModelConfig:
    """Network size. Small defaults keep CPU iteration fast; bump these on GPU."""

    num_blocks: int = 6
    channels: int = 96
    value_hidden: int = 128


@dataclass
class Config:
    device: str = field(default_factory=get_device)

    # Path to a UCI Stockfish binary. Override with STOCKFISH_PATH.
    stockfish_path: str = field(
        default_factory=lambda: os.environ.get("STOCKFISH_PATH", "stockfish")
    )

    # Optional Polyglot opening book (.bin) and Syzygy endgame tablebase directory.
    # Used only for actual play/analysis, not training. Empty = disabled.
    opening_book_path: str = field(
        default_factory=lambda: os.environ.get("OPENING_BOOK", "")
    )
    syzygy_path: str = field(default_factory=lambda: os.environ.get("SYZYGY_PATH", ""))

    model: ModelConfig = field(default_factory=ModelConfig)

    # Directories (created on demand).
    models_dir: str = field(default_factory=lambda: os.environ.get("CHESSAI_MODELS", "models"))
    data_dir: str = field(default_factory=lambda: os.environ.get("CHESSAI_DATA", "data_store"))
    logs_dir: str = field(default_factory=lambda: os.environ.get("CHESSAI_LOGS", "logs"))

    # Search defaults.
    mcts_simulations: int = 160
    mcts_batch_size: int = 32
    mcts_virtual_loss: int = 3
    nn_cache_size: int = 50_000
    infer_backend: str = "torch"  # torch | onnx | onnx-int8 | tensorrt
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    use_torch_compile: bool = False  # set True or CHESSAI_COMPILE=1 on CUDA

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.data_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


# A module-level default instance for convenience.
CONFIG = Config()
