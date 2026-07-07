"""Pluggable inference backends for MCTS (PyTorch, ONNX Runtime, optional TensorRT EP).

Set ``CHESSAI_INFER`` to choose the runtime:
  - ``torch`` (default) — eager PyTorch; enable compile with ``CHESSAI_COMPILE=1``
  - ``onnx`` — ONNX Runtime float32 (good CPU/GPU win after one-time export)
  - ``onnx-int8`` — dynamic int8 weights (best on CPU)
  - ``tensorrt`` — ONNX Runtime TensorRT EP (NVIDIA GPU only; needs tensorrt pip)

Export once:
  python -m cli export-onnx --checkpoint models/supervised_big.pt --int8
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from engine.config import Config
from engine.encoding import NUM_PLANES
from engine.model import ChessNet, build_model, load_checkpoint

BatchOut = Tuple[np.ndarray, np.ndarray]  # logits (N, P), values (N,)


class InferenceRunner(ABC):
    """Batch inference used by MCTS (planes in -> logits + values out)."""

    model: Optional[ChessNet] = None

    @abstractmethod
    def eval_batch(self, planes: np.ndarray) -> BatchOut:
        """``planes`` is float32 array of shape (N, 18, 8, 8)."""


class TorchRunner(InferenceRunner):
    def __init__(self, model: ChessNet, device: str):
        import torch

        self.model = model
        self.device = device
        self._torch = torch

    def eval_batch(self, planes: np.ndarray) -> BatchOut:
        x = self._torch.from_numpy(planes).to(self.device)
        with self._torch.no_grad():
            logits, values = self.model(x)
        return logits.detach().cpu().numpy(), values.detach().cpu().numpy()


class ONNXRunner(InferenceRunner):
    def __init__(self, onnx_path: str, providers: Optional[List] = None):
        import onnxruntime as ort

        if providers is None:
            providers = ort_providers()
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def eval_batch(self, planes: np.ndarray) -> BatchOut:
        out = self.session.run(None, {self.input_name: planes})
        return out[0], out[1]


def ort_providers() -> List:
    """Pick ONNX Runtime execution providers from env + hardware."""
    import onnxruntime as ort

    want = os.environ.get("CHESSAI_INFER", "onnx").lower()
    available = set(ort.get_available_providers())

    if want == "tensorrt" and "TensorrtExecutionProvider" in available:
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if "CUDAExecutionProvider" in available and os.environ.get("CHESSAI_DEVICE", "") == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def onnx_paths_for_checkpoint(checkpoint: str, int8: bool = False) -> Tuple[str, str]:
    """Return (fp32_onnx_path, int8_onnx_path) beside the checkpoint."""
    base = os.path.splitext(checkpoint)[0]
    return base + ".onnx", base + ".int8.onnx"


def export_onnx(
    checkpoint: str,
    out_path: Optional[str] = None,
    int8: bool = False,
    opset: int = 17,
) -> str:
    """Export a ``.pt`` checkpoint to ONNX; optionally quantize to int8."""
    import torch

    fp32_path, int8_path = onnx_paths_for_checkpoint(checkpoint)
    if out_path is None:
        out_path = int8_path if int8 else fp32_path

    model, _ = load_checkpoint(checkpoint, device="cpu", compile_model=False)
    model.eval()

    dummy = torch.randn(1, NUM_PLANES, 8, 8)
    export_kw = dict(
        input_names=["planes"],
        output_names=["policy", "value"],
        dynamic_axes={
            "planes": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=opset,
    )
    # Legacy exporter is more portable across torch versions / Windows consoles.
    try:
        torch.onnx.export(model, dummy, fp32_path, dynamo=False, **export_kw)
    except TypeError:
        torch.onnx.export(model, dummy, fp32_path, **export_kw)

    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Dynamic quant on Conv can fail when ONNX keeps weights as graph nodes
        # (common with some torch exporters). Quantizing Gemm/MatMul still helps
        # because the large policy FC layer dominates inference cost.
        try:
            quantize_dynamic(
                fp32_path,
                int8_path,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
        except Exception:
            quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
        return int8_path
    return fp32_path


def resolve_infer_backend(
    config: Optional[Config] = None, checkpoint: Optional[str] = None
) -> str:
    """Pick inference backend: env override, then auto-detect ONNX beside checkpoint."""
    env = os.environ.get("CHESSAI_INFER")
    if env:
        return env.lower()
    config = config or Config()
    if checkpoint:
        fp32, int8 = onnx_paths_for_checkpoint(checkpoint, int8=True)
        if os.path.exists(int8):
            return "onnx-int8"
        if os.path.exists(fp32):
            return "onnx"
    return config.infer_backend.lower()


def create_inference_runner(
    config: Optional[Config] = None,
    checkpoint: Optional[str] = None,
    model: Optional[ChessNet] = None,
) -> InferenceRunner:
    """Build the inference backend selected by ``CHESSAI_INFER`` / config."""
    config = config or Config()
    infer = resolve_infer_backend(config, checkpoint)

    if infer in ("onnx", "onnx-int8", "int8", "tensorrt"):
        int8 = infer in ("onnx-int8", "int8")
        onnx_env = os.environ.get("CHESSAI_ONNX")
        if onnx_env:
            path = onnx_env
        elif checkpoint:
            fp32, quant = onnx_paths_for_checkpoint(checkpoint, int8=True)
            path = quant if int8 else fp32
            if not os.path.exists(path) and checkpoint and os.path.exists(checkpoint):
                export_onnx(checkpoint, int8=int8)
        else:
            raise RuntimeError("ONNX inference requires CHESSAI_ONNX or a checkpoint path to export from.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return ONNXRunner(path)

    # PyTorch (default).
    if model is None:
        if not checkpoint or not os.path.exists(checkpoint):
            model = build_model(config.model, device=config.device, compile_model=_want_compile(config))
        else:
            model, _ = load_checkpoint(checkpoint, device=config.device, compile_model=_want_compile(config))
    elif _want_compile(config):
        model = _maybe_compile(model, config.device)
    return TorchRunner(model, config.device)


def _want_compile(config: Config) -> bool:
    if config.use_torch_compile:
        return True
    return os.environ.get("CHESSAI_COMPILE", "").lower() in ("1", "true", "yes")


def _maybe_compile(model: ChessNet, device: str) -> ChessNet:
    from engine.model import _maybe_torch_compile

    if not _want_compile(Config()):
        return model
    return _maybe_torch_compile(model, device)
