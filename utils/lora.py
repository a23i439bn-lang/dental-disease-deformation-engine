from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

''' LoRA（Low-Rank Adaptation）を実装するモジュールです。'''
def sanitize_disease_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)

''' 疾患名を正規化するためのエイリアスとプリセットの定義 '''
@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0


class MultiDiseaseLoRALinear(nn.Module):
    """
    LoRA wrapper for cross-attention linear layers.

    Research note:
    Base diffusion weights remain frozen; disease-specific residuals are learned
    independently, which improves safety and lets us add new diseases cheaply.
    """

    def __init__(self, base_layer: nn.Linear, disease_names: list[str], config: LoRAConfig) -> None:
        super().__init__()
        self.base = base_layer
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.down = nn.ParameterDict()
        self.up = nn.ParameterDict()
        for disease in disease_names:
            key = sanitize_disease_name(disease)
            self.down[key] = nn.Parameter(torch.randn(config.rank, self.in_features) * 0.01)
            self.up[key] = nn.Parameter(torch.zeros(self.out_features, config.rank))
        self.active_scales: dict[str, float] = {}

    def set_active_scales(self, scales: dict[str, float]) -> None:
        self.active_scales = {sanitize_disease_name(key): float(value) for key, value in scales.items()}

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.config.rank}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.active_scales:
            return base_out
        update = torch.zeros_like(base_out)
        scaling = self.config.alpha / max(self.config.rank, 1)
        for key, scale in self.active_scales.items():
            if key not in self.down:
                continue
            low_rank = F.linear(x, self.down[key])
            low_rank = F.linear(low_rank, self.up[key])
            update = update + low_rank * (scale * scaling)
        return base_out + update

''' LoRAを管理するクラス '''
class DiseaseLoRAManager:
    def __init__(self, disease_names: list[str], config: LoRAConfig) -> None:
        self.disease_names = disease_names
        self.config = config
        self._wrapped_modules: dict[str, MultiDiseaseLoRALinear] = {}

    @staticmethod
    def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def inject_into_unet(self, unet: nn.Module) -> None:
        target_names = []
        for name, module in unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(token in name for token in ("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")):
                target_names.append(name)
        for name in target_names:
            parent, child_name = self._resolve_parent(unet, name)
            wrapped = MultiDiseaseLoRALinear(getattr(parent, child_name), self.disease_names, self.config)
            setattr(parent, child_name, wrapped)
            self._wrapped_modules[name] = wrapped

    def set_active_diseases(self, disease_scales: dict[str, float]) -> None:
        for module in self._wrapped_modules.values():
            module.set_active_scales(disease_scales)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self._wrapped_modules.values():
            params.extend(list(module.down.parameters()))
            params.extend(list(module.up.parameters()))
        return params

    def save_per_disease(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for disease in self.disease_names:
            key = sanitize_disease_name(disease)
            payload = {
                "disease": disease,
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "layers": {},
            }
            for layer_name, module in self._wrapped_modules.items():
                payload["layers"][layer_name] = {
                    "down": module.down[key].detach().cpu(),
                    "up": module.up[key].detach().cpu(),
                }
            torch.save(payload, output_path / f"lora_{key}.pt")

    def load_per_disease(self, paths: list[str | Path]) -> None:
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            disease = sanitize_disease_name(payload["disease"])
            for layer_name, weights in payload["layers"].items():
                if layer_name not in self._wrapped_modules:
                    continue
                module = self._wrapped_modules[layer_name]
                module.down[disease].data.copy_(weights["down"])
                module.up[disease].data.copy_(weights["up"])
