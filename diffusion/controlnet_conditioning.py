from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ControlConditionConfig:
    image_size: int = 256
    conditioning_channels: int = 3
    hidden_channels: int = 32
    max_flow_norm: float = 32.0


class ControlConditionAdapter(nn.Module):
    """
    Builds a stable ControlNet conditioning tensor from geometry + texture + severity.

    Research note:
    We do not pass raw tensors directly into ControlNet. Instead, we normalize them into
    a compact control image so the downstream diffusion model receives bounded signals.
    This substantially reduces rendered image collapse.
    """

    def __init__(self, config: ControlConditionConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = nn.Sequential(
            nn.Conv2d(7, config.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, config.hidden_channels),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, config.conditioning_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        dense_flow: torch.Tensor,
        landmark_heatmap: torch.Tensor,
        texture_residual: torch.Tensor,
        severity: torch.Tensor,
    ) -> torch.Tensor:
        if dense_flow.ndim != 4 or dense_flow.shape[1] != 2:
            raise ValueError(f"dense_flow must have shape (B, 2, H, W), got {tuple(dense_flow.shape)}")
        if landmark_heatmap.ndim != 4 or landmark_heatmap.shape[1] != 1:
            raise ValueError(f"landmark_heatmap must have shape (B, 1, H, W), got {tuple(landmark_heatmap.shape)}")
        if texture_residual.ndim != 4 or texture_residual.shape[1] != 3:
            raise ValueError(f"texture_residual must have shape (B, 3, H, W), got {tuple(texture_residual.shape)}")
        if severity.ndim == 1:
            severity = severity.unsqueeze(-1)
        if severity.ndim != 2 or severity.shape[-1] != 1:
            raise ValueError(f"severity must have shape (B, 1), got {tuple(severity.shape)}")

        batch_size, _, height, width = dense_flow.shape
        severity_map = severity[:, :, None, None].expand(batch_size, 1, height, width)
        flow = torch.nan_to_num(dense_flow, nan=0.0, posinf=0.0, neginf=0.0)
        flow = flow / self.config.max_flow_norm
        flow = flow.clamp(-1.0, 1.0)
        heatmap = torch.nan_to_num(landmark_heatmap, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        residual = torch.nan_to_num(texture_residual, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        severity_map = severity_map.clamp(0.0, 1.0)
        conditioning = torch.cat([flow, heatmap, residual, severity_map], dim=1)
        conditioning = self.projector(conditioning)
        conditioning = torch.nan_to_num(conditioning, nan=0.0, posinf=0.0, neginf=0.0)
        return (conditioning * 0.5 + 0.5).clamp(0.0, 1.0)


def build_landmark_heatmap(
    source_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    image_size: tuple[int, int],
    sigma: float = 6.0,
) -> torch.Tensor:
    if source_landmarks.shape != target_landmarks.shape:
        raise ValueError("source_landmarks and target_landmarks must have identical shapes.")
    batch_size, num_points, _ = source_landmarks.shape
    height, width = image_size
    ys, xs = torch.meshgrid(
        torch.arange(height, device=source_landmarks.device, dtype=source_landmarks.dtype),
        torch.arange(width, device=source_landmarks.device, dtype=source_landmarks.dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1).view(1, 1, height, width, 2)
    target = target_landmarks[:, :, None, None, :]
    dist2 = ((grid - target) ** 2).sum(dim=-1)
    heat = torch.exp(-dist2 / (2.0 * sigma ** 2)).sum(dim=1, keepdim=True)
    heat = heat / heat.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
