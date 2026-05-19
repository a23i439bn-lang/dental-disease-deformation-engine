from __future__ import annotations

"""変形後の画像へ局所的なテクスチャ変化を足すモデル。

虫歯や歯肉炎のように「形」だけでなく「見た目」も変える必要がある
疾患を扱うための支流。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TextureBranchConfig:
    in_channels: int = 3
    disease_dim: int = 256
    base_channels: int = 32
    out_channels: int = 3


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConvFiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.film1 = FiLM(cond_dim, out_channels)
        self.film2 = FiLM(cond_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.film1(self.norm1(x), cond)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.film2(self.norm2(x), cond)
        x = F.silu(x + residual)
        return x


class TextureBranch(nn.Module):
    def __init__(self, config: TextureBranchConfig) -> None:
        super().__init__()
        c = config.base_channels
        self.enc1 = ConvFiLMBlock(config.in_channels, c, config.disease_dim)
        self.enc2 = ConvFiLMBlock(c, c * 2, config.disease_dim)
        self.mid = ConvFiLMBlock(c * 2, c * 2, config.disease_dim)
        self.dec1 = ConvFiLMBlock(c * 3, c, config.disease_dim)
        self.out = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, config.out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, warped_image: torch.Tensor, disease_embed: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(warped_image, disease_embed)
        x2 = self.enc2(F.avg_pool2d(x1, kernel_size=2), disease_embed)
        xm = self.mid(x2, disease_embed)
        xu = F.interpolate(xm, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        xd = self.dec1(torch.cat([xu, x1], dim=1), disease_embed)
        return self.out(xd)
