from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from diffusion.pipeline import DenseWarpLayer, DiseaseAwareDiffusionPipeline, DiseaseDiffusionRenderer, RendererConfig
from models.deformation_policy import DeformationPolicyConfig, DeformationPolicyNetwork
from models.disease_encoder import DiseaseEncoder, DiseaseEncoderConfig
from models.severity_policy import SeverityActor, SeverityPolicyConfig
from models.texture_branch import TextureBranch, TextureBranchConfig

''' デバイス名を解決する関数 '''
def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device_name

''' モジュールのパラメータを凍結する関数 '''
def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()
    return module

''' 研究用モジュールを構築する関数 '''
def build_research_modules(config: dict[str, Any], device: str) -> dict[str, nn.Module]:
    disease_dim = int(config["model"]["disease_dim"])
    diseases = list(config["data"]["diseases"])
    texture_branch = TextureBranch(
        TextureBranchConfig(
            disease_dim=disease_dim,
            base_channels=int(config["model"]["texture_base_channels"]),
        )
    )
    renderer = DiseaseDiffusionRenderer(
        RendererConfig(
            model_id=config["diffusion"]["model_id"],
            device=device,
            dtype="float16" if device == "cuda" else "float32",
            guidance_scale=float(config["diffusion"]["guidance_scale"]),
            strength=float(config["diffusion"]["strength"]),
            num_inference_steps=int(config["diffusion"]["num_inference_steps"]),
            cross_attention_dim=int(config["diffusion"]["cross_attention_dim"]),
            num_disease_tokens=int(config["diffusion"]["num_disease_tokens"]),
            disease_dim=disease_dim,
            enable_lora=bool(config["diffusion"].get("enable_lora", False)),
            lora_rank=int(config["diffusion"].get("lora_rank", 4)),
            fallback_only=bool(config["diffusion"].get("fallback_only", False)),
            conditioning_channels=int(config["diffusion"].get("conditioning_channels", 3)),
            controlnet_scale=float(config["diffusion"].get("controlnet_scale", 1.0)),
        ),
        texture_branch=texture_branch,
        disease_names=diseases,
    )
    modules: dict[str, nn.Module] = {
        "disease_encoder": DiseaseEncoder(
            DiseaseEncoderConfig(
                num_diseases=len(diseases),
                embed_dim=disease_dim,
            )
        ).to(device),
        "deformation_policy": DeformationPolicyNetwork(
            DeformationPolicyConfig(
                disease_dim=disease_dim,
                hidden_dim=int(config["model"]["deformation_hidden_dim"]),
            )
        ).to(device),
        "texture_branch": texture_branch.to(device),
        "severity_actor": SeverityActor(
            SeverityPolicyConfig(
                disease_dim=disease_dim,
                hidden_dim=int(config["rl"].get("actor_hidden_dim", 256)),
            )
        ).to(device),
        "pipeline": DiseaseAwareDiffusionPipeline(
            warp_layer=DenseWarpLayer(sigma=float(config["model"]["warp_sigma"])),
            texture_branch=texture_branch,
            renderer=renderer,
        ).to(device),
    }
    return modules

''' モジュールのパラメータを収集する関数 '''
def collect_unique_trainable_params(*modules: nn.Module) -> list[torch.nn.Parameter]:
    unique_params: dict[int, torch.nn.Parameter] = {}
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                unique_params[id(param)] = param
    return list(unique_params.values())
