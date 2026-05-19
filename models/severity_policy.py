from __future__ import annotations

"""severity の扱いを補助するモデル。

疾患の強さをどう反映させるかを整理するための部品で、
強度制御を学習化したいときの拡張点になる。
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeverityPolicyConfig:
    image_channels: int = 3
    disease_dim: int = 256
    hidden_dim: int = 256
    min_std: float = 0.05


class SeverityActor(nn.Module):
    """
    Actor network for automatic severity optimization.

    Research note:
    We optimize severity with detached rewards only. This avoids unstable gradients
    through diffusion while still letting the actor discover better severity schedules.
    """

    def __init__(self, config: SeverityPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = nn.Sequential(
            nn.Conv2d(config.image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + config.disease_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, source_image: torch.Tensor, disease_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_feat = self.image_encoder(source_image).flatten(1)
        fused = self.fusion(torch.cat([image_feat, disease_embed], dim=-1))
        mean = torch.sigmoid(self.mean_head(fused))
        std = self.log_std.exp().clamp_min(self.config.min_std).expand_as(mean)
        return mean, std

    def sample(self, source_image: torch.Tensor, disease_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, std = self(source_image, disease_embed)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        severity = raw.clamp(0.0, 1.0)
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        return {"severity": severity, "log_prob": log_prob, "mean": mean, "std": std}


class PPOUpdater:
    def __init__(self, clip_eps: float = 0.2) -> None:
        self.clip_eps = clip_eps

    def loss(
        self,
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        advantage = reward - reward.mean()
        advantage = advantage / advantage.std().clamp_min(1e-6)
        ratio = (new_log_prob - old_log_prob).exp()
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
        return -torch.min(unclipped, clipped).mean()
