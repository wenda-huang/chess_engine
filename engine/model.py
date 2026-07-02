"""The policy+value network (a small AlphaZero-style ResNet)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.config import ModelConfig
from engine.encoding import NUM_PLANES, POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ChessNet(nn.Module):
    """Input: (B, 18, 8, 8). Outputs: policy logits (B, POLICY_SIZE) and value (B,)."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        c = config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(c) for _ in range(config.num_blocks)])

        # Policy head.
        self.policy_conv = nn.Conv2d(c, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_SIZE)

        # Value head.
        self.value_conv = nn.Conv2d(c, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


def build_model(config: ModelConfig | None = None, device: str = "cpu") -> ChessNet:
    model = ChessNet(config)
    model.to(device)
    return model


def save_checkpoint(path: str, model: ChessNet, meta: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": vars(model.config),
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[ChessNet, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt.get("model_config", {}))
    model = build_model(cfg, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})
