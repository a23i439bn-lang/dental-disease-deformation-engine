from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPTokenizer
except Exception:  # pragma: no cover
    CLIPModel = None  # type: ignore[assignment]
    CLIPTokenizer = None  # type: ignore[assignment]


@dataclass
class CLIPLossConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embed_dim: int = 512
    healthy_prompt: str = "healthy mouth"
    image_size: int = 224


class FallbackTextEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)

    def forward(self, prompts: Iterable[str], device: torch.device) -> torch.Tensor:
        rows = []
        for prompt in prompts:
            codepoints = [ord(ch) % 256 for ch in prompt[:128]] or [0]
            tokens = torch.tensor(codepoints, device=device, dtype=torch.long)
            rows.append(self.embedding(tokens).mean(dim=0))
        return F.normalize(torch.stack(rows, dim=0), dim=-1)


class FallbackImageEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image).flatten(1)
        return F.normalize(self.proj(features), dim=-1)


class CLIPDiseaseLoss(nn.Module):
    def __init__(self, config: CLIPLossConfig, device: str) -> None:
        super().__init__()
        self.config = config
        self.device_name = device
        self.model = None
        self.tokenizer = None
        self.fallback_text = FallbackTextEncoder(config.embed_dim)
        self.fallback_image = FallbackImageEncoder(config.embed_dim)
        self._load_clip()

    def _load_clip(self) -> None:
        if CLIPModel is None or CLIPTokenizer is None:
            return
        try:
            self.model = CLIPModel.from_pretrained(self.config.model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)
        except Exception:
            self.model = None
            self.tokenizer = None

    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            return self.fallback_text(prompts, device)
        tokens = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        text_features = self.model.get_text_features(**tokens)
        return F.normalize(text_features, dim=-1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return self.fallback_image(image)
        image = F.interpolate(image, size=(self.config.image_size, self.config.image_size), mode="bilinear", align_corners=False)
        image = (image.clamp(0.0, 1.0) - 0.5) / 0.5
        image_features = self.model.get_image_features(pixel_values=image)
        return F.normalize(image_features, dim=-1)

    def forward(self, image: torch.Tensor, disease_prompts: list[str]) -> torch.Tensor:
        device = image.device
        image_features = self.encode_image(image)
        disease_features = self.encode_text(disease_prompts, device)
        healthy_features = self.encode_text([self.config.healthy_prompt] * len(disease_prompts), device)
        loss_disease = 1.0 - F.cosine_similarity(image_features, disease_features, dim=-1)
        loss_healthy = F.cosine_similarity(image_features, healthy_features, dim=-1)
        return (loss_disease + loss_healthy).mean()
