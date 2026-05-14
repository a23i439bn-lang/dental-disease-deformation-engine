from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DiseaseEncoderConfig:
    num_diseases: int
    embed_dim: int = 256
    severity_dim: int = 64
    hidden_dim: int = 256
    use_layernorm: bool = True


class DiseaseEncoder(nn.Module):
    """Encodes multi-hot disease labels and continuous severity into a single embedding."""

    def __init__(self, config: DiseaseEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_diseases, config.embed_dim)
        self.severity_mlp = nn.Sequential(
            nn.Linear(1, config.severity_dim),
            nn.SiLU(),
            nn.Linear(config.severity_dim, config.embed_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )
        self.norm = nn.LayerNorm(config.embed_dim) if config.use_layernorm else nn.Identity()

    def forward(self, multi_hot: torch.Tensor, severity: torch.Tensor | None = None) -> torch.Tensor:
        if multi_hot.ndim != 2:
            raise ValueError(f"Expected multi_hot to have shape (B, N), got {tuple(multi_hot.shape)}")
        multi_hot = multi_hot.float()
        denom = multi_hot.sum(dim=1, keepdim=True).clamp_min(1.0)
        disease_embed = multi_hot @ self.embedding.weight
        disease_embed = disease_embed / denom

        if severity is None:
            severity = torch.zeros(multi_hot.shape[0], 1, device=multi_hot.device, dtype=multi_hot.dtype)
        if severity.ndim == 1:
            severity = severity.unsqueeze(-1)
        severity_embed = self.severity_mlp(severity.float())

        fused = self.fusion(torch.cat([disease_embed, severity_embed], dim=-1))
        return self.norm(fused)
