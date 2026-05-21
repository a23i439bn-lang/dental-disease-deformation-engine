from __future__ import annotations
# このファイルの役割:
# ランドマークの変形量を予測するモデルです。
# 疾患 embedding と顔ランドマークから、各点の移動量を計算します。

"""ランドマークをどの方向へどれだけ動かすかを予測するモデル。

疾患embeddingを受け取り、468点の各ランドマークに対する delta を返す。
「形を疾患化する」担当の主役モデル。
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_landmark_edge_index(num_landmarks: int = 468) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    for idx in range(num_landmarks - 1):
        edges.append((idx, idx + 1))
        edges.append((idx + 1, idx))
    mouth_ring = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
        61, 78, 95, 88, 178, 87, 14, 317, 402, 318,
        324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 78,
    ]
    for src, dst in zip(mouth_ring[:-1], mouth_ring[1:]):
        edges.append((src, dst))
        edges.append((dst, src))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@dataclass
class DeformationPolicyConfig:
    num_landmarks: int = 468
    landmark_dim: int = 2
    disease_dim: int = 256
    hidden_dim: int = 192
    num_layers: int = 4
    dropout: float = 0.1
    roi_boost: float = 1.8


class GraphConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neighbor_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        batch_size, num_nodes, feat_dim = x.shape
        agg = torch.zeros(batch_size, num_nodes, feat_dim, device=x.device, dtype=x.dtype)
        agg.index_add_(1, dst, x[:, src, :])
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0).view(1, num_nodes, 1)
        out = self.self_proj(x) + self.neighbor_proj(agg)
        out = self.dropout(F.silu(self.norm(out)))
        return out


class DeformationPolicyNetwork(nn.Module):
    def __init__(self, config: DeformationPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("edge_index", build_landmark_edge_index(config.num_landmarks), persistent=False)

        self.input_proj = nn.Linear(config.landmark_dim + config.disease_dim + 2, config.hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConvBlock(config.hidden_dim, config.hidden_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )
        self.mouth_mask = self._build_mouth_mask(config.num_landmarks)
        self.debug_landmarks = os.environ.get("DEBUG_LANDMARKS", "0") == "1"

    @staticmethod
    def _build_mouth_mask(num_landmarks: int) -> torch.Tensor:
        mask = torch.zeros(num_landmarks, dtype=torch.float32)
        mouth_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        ]
        mask[mouth_indices] = 1.0
        return mask.view(1, num_landmarks, 1)

    def forward(self, landmarks: torch.Tensor, disease_embed: torch.Tensor) -> torch.Tensor:
        if landmarks.ndim != 3:
            raise ValueError(f"Expected landmarks to have shape (B, 468, 2), got {tuple(landmarks.shape)}")
        batch_size, num_nodes, _ = landmarks.shape
        if num_nodes != self.config.num_landmarks:
            raise ValueError(f"Expected {self.config.num_landmarks} landmarks, got {num_nodes}")

        disease_expand = disease_embed[:, None, :].expand(batch_size, num_nodes, -1)
        node_coords = torch.linspace(-1.0, 1.0, steps=num_nodes, device=landmarks.device, dtype=landmarks.dtype)
        node_pos = torch.stack([node_coords, torch.cos(node_coords * torch.pi)], dim=-1)
        node_pos = node_pos.unsqueeze(0).expand(batch_size, -1, -1)

        x = torch.cat([landmarks, disease_expand, node_pos], dim=-1)
        x = F.silu(self.input_proj(x))
        for layer in self.layers:
            x = x + layer(x, self.edge_index)
        delta = self.output_head(x)
        if self.debug_landmarks:
            print(f"[deformation_policy] landmarks.shape={tuple(landmarks.shape)}")
            print(f"[deformation_policy] delta.abs().mean()={delta.abs().mean().item():.6f}")
        mouth_mask = self.mouth_mask.to(delta.device, delta.dtype)
        return delta * (1.0 + mouth_mask * (self.config.roi_boost - 1.0))
