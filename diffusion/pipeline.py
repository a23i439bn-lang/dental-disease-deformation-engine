from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
except Exception:  # pragma: no cover
    ControlNetModel = None  # type: ignore[assignment]
    StableDiffusionControlNetImg2ImgPipeline = None  # type: ignore[assignment]

from diffusion.controlnet_conditioning import ControlConditionAdapter, ControlConditionConfig, build_landmark_heatmap
from diffusion.disease_attention import (
    DiseaseAttentionConfig,
    DiseaseCrossAttentionProcessor,
    DiseaseTokenProjector,
    run_cross_attention_safety_checks,
)
from models.texture_branch import TextureBranch
from utils.lora import DiseaseLoRAManager, LoRAConfig


@dataclass
class RendererConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    dtype: str = "float32"
    guidance_scale: float = 6.0
    strength: float = 0.35
    num_inference_steps: int = 20
    cross_attention_dim: int = 768
    num_disease_tokens: int = 4
    disease_dim: int = 256
    latent_scale_norm: float = 0.18215
    max_condition_norm: float = 24.0
    enable_lora: bool = False
    lora_rank: int = 4
    fallback_only: bool = False
    conditioning_channels: int = 3
    controlnet_scale: float = 1.0


def clamp_and_normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = torch.nan_to_num(image.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    mean = image.mean(dim=(2, 3), keepdim=True)
    std = image.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    return ((image - mean) / std * 0.25 + 0.5).clamp(0.0, 1.0)


def is_render_tensor_valid(image: torch.Tensor) -> bool:
    if torch.isnan(image).any() or torch.isinf(image).any():
        return False
    dynamic_range = float((image.max() - image.min()).detach().cpu())
    std = float(image.std().detach().cpu())
    return dynamic_range > 1e-3 and std > 1e-4


def apply_disease_heuristic_boost(
    image: torch.Tensor,
    active_diseases: list[str],
    severity: torch.Tensor,
) -> torch.Tensor:
    """
    Research note:
    This is a safety-net for zero-shot / no-checkpoint operation.
    It ensures disease evidence is visually present even before end-to-end training.
    """
    boosted = image.clone()
    severity_scale = float(severity.mean().detach().cpu())
    batch, channels, height, width = boosted.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=boosted.device, dtype=boosted.dtype),
        torch.linspace(-1.0, 1.0, width, device=boosted.device, dtype=boosted.dtype),
        indexing="ij",
    )
    mouth_mask = torch.exp(-((xx / 0.35) ** 2 + ((yy - 0.35) / 0.16) ** 2)).unsqueeze(0).unsqueeze(0)
    mouth_mask = mouth_mask.expand(batch, 1, -1, -1)

    if "虫歯" in active_diseases or "陌ｫ豁ｯ" in active_diseases:
        darken = 0.30 * severity_scale
        stain = torch.cat(
            [
                0.12 * mouth_mask,
                -0.18 * mouth_mask,
                -0.24 * mouth_mask,
            ],
            dim=1,
        )
        boosted = boosted * (1.0 - darken * mouth_mask) + stain
    if "出っ歯" in active_diseases or "蜃ｺ縺｣豁ｯ" in active_diseases:
        shift = max(1, int(8 * severity_scale))
        boosted = torch.roll(boosted, shifts=shift, dims=-1) * mouth_mask + boosted * (1.0 - mouth_mask)
    if "すきっ歯" in active_diseases or "縺吶″縺｣豁ｯ" in active_diseases:
        center = width // 2
        gap = max(2, int(6 * severity_scale))
        boosted[:, :, :, max(0, center - gap):min(width, center + gap)] = 1.0

    return boosted.clamp(0.0, 1.0)


class DenseWarpLayer(nn.Module):
    def __init__(self, sigma: float = 28.0) -> None:
        super().__init__()
        self.sigma = sigma

    def landmarks_to_flow(
        self,
        source_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if source_landmarks.shape != target_landmarks.shape:
            raise ValueError("source_landmarks and target_landmarks must match.")
        height, width = image_size
        ys, xs = torch.meshgrid(
            torch.arange(height, device=source_landmarks.device, dtype=source_landmarks.dtype),
            torch.arange(width, device=source_landmarks.device, dtype=source_landmarks.dtype),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1).view(1, height, width, 1, 2)
        src = source_landmarks[:, None, None, :, :]
        disp = (target_landmarks - source_landmarks)[:, None, None, :, :]
        dist2 = ((grid - src) ** 2).sum(dim=-1)
        weights = torch.exp(-dist2 / (2.0 * self.sigma ** 2))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        flow = (weights.unsqueeze(-1) * disp).sum(dim=3)
        return torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0).permute(0, 3, 1, 2)

    def forward(
        self,
        image: torch.Tensor,
        source_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = image.shape
        flow = self.landmarks_to_flow(source_landmarks, target_landmarks, (height, width))
        norm_flow = torch.zeros_like(flow)
        norm_flow[:, 0] = flow[:, 0] / max(width - 1, 1) * 2.0
        norm_flow[:, 1] = flow[:, 1] / max(height - 1, 1) * 2.0
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        sampling_grid = torch.stack([xs, ys], dim=-1).unsqueeze(0) - norm_flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(image, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return clamp_and_normalize_image(warped), flow


class DiseaseDiffusionRenderer(nn.Module):
    def __init__(
        self,
        config: RendererConfig,
        texture_branch: TextureBranch,
        disease_names: list[str],
    ) -> None:
        super().__init__()
        self.config = config
        self.texture_branch = texture_branch
        self.disease_names = disease_names
        self.attn_config = DiseaseAttentionConfig(
            disease_dim=config.disease_dim,
            cross_attention_dim=config.cross_attention_dim,
            num_disease_tokens=config.num_disease_tokens,
        )
        self.projector = DiseaseTokenProjector(self.attn_config)
        self.control_adapter = ControlConditionAdapter(
            ControlConditionConfig(conditioning_channels=config.conditioning_channels)
        )
        self.pipeline = None
        self.controlnet = None
        self.attn_processors: list[DiseaseCrossAttentionProcessor] = []
        self.lora_manager: DiseaseLoRAManager | None = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        if self.config.fallback_only or StableDiffusionControlNetImg2ImgPipeline is None or ControlNetModel is None:
            self.pipeline = None
            self.controlnet = None
            self.attn_processors = []
            self.lora_manager = None
            return
        dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=dtype,
            )
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config.model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )
            processors: dict[str, DiseaseCrossAttentionProcessor] = {}
            for name in pipe.unet.attn_processors.keys():
                processor = DiseaseCrossAttentionProcessor(self.projector)
                processors[name] = processor
                self.attn_processors.append(processor)
            pipe.unet.set_attn_processor(processors)
            self.controlnet = pipe.controlnet
            if self.config.enable_lora:
                self.lora_manager = DiseaseLoRAManager(
                    disease_names=self.disease_names,
                    config=LoRAConfig(rank=self.config.lora_rank, alpha=float(self.config.lora_rank)),
                )
                self.lora_manager.inject_into_unet(pipe.unet)
            pipe.to(self.config.device)
            self.pipeline = pipe
        except Exception:
            self.pipeline = None
            self.controlnet = None
            self.attn_processors = []
            self.lora_manager = None

    def renderer_state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "renderer_config": asdict(self.config),
            "projector_state": self.projector.state_dict(),
            "control_adapter_state": self.control_adapter.state_dict(),
            "has_pipeline": self.pipeline is not None,
        }
        if self.pipeline is not None:
            payload["unet_state"] = self.pipeline.unet.state_dict()
            payload["controlnet_state"] = self.pipeline.controlnet.state_dict()
        return payload

    def load_renderer_state(self, renderer_state: dict[str, Any], strict: bool = True) -> dict[str, list[str]]:
        mismatches = {"missing_keys": [], "unexpected_keys": []}
        projector_result = self.projector.load_state_dict(renderer_state["projector_state"], strict=strict)
        adapter_result = self.control_adapter.load_state_dict(renderer_state["control_adapter_state"], strict=strict)
        mismatches["missing_keys"].extend(projector_result.missing_keys)
        mismatches["unexpected_keys"].extend(projector_result.unexpected_keys)
        mismatches["missing_keys"].extend([f"control_adapter:{k}" for k in adapter_result.missing_keys])
        mismatches["unexpected_keys"].extend([f"control_adapter:{k}" for k in adapter_result.unexpected_keys])
        if self.pipeline is not None:
            unet_result = self.pipeline.unet.load_state_dict(renderer_state["unet_state"], strict=strict)
            control_result = self.pipeline.controlnet.load_state_dict(renderer_state["controlnet_state"], strict=strict)
            mismatches["missing_keys"].extend([f"unet:{k}" for k in unet_result.missing_keys])
            mismatches["unexpected_keys"].extend([f"unet:{k}" for k in unet_result.unexpected_keys])
            mismatches["missing_keys"].extend([f"controlnet:{k}" for k in control_result.missing_keys])
            mismatches["unexpected_keys"].extend([f"controlnet:{k}" for k in control_result.unexpected_keys])
        elif strict and renderer_state.get("has_pipeline", False):
            raise RuntimeError("Checkpoint expects ControlNet pipeline but current environment is fallback-only.")
        print(f"[renderer] missing_keys={mismatches['missing_keys']}")
        print(f"[renderer] unexpected_keys={mismatches['unexpected_keys']}")
        return mismatches

    def set_lora_scales(self, disease_names: list[str], severity: torch.Tensor) -> None:
        if self.lora_manager is None:
            return
        mean_scale = float(severity.mean().detach().cpu())
        scales = {name: mean_scale for name in disease_names}
        self.lora_manager.set_active_diseases(scales)

    def _stabilize_condition(self, disease_embed: torch.Tensor) -> torch.Tensor:
        run_cross_attention_safety_checks(
            disease_embed=disease_embed,
            expected_disease_dim=self.config.disease_dim,
            expected_num_tokens=self.config.num_disease_tokens,
        )
        disease_embed = torch.nan_to_num(disease_embed, nan=0.0, posinf=0.0, neginf=0.0)
        norms = disease_embed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if float(norms.max().detach().cpu()) > self.config.max_condition_norm:
            disease_embed = disease_embed / norms * self.config.max_condition_norm
        return disease_embed

    def _fallback_render(self, image: torch.Tensor, disease_embed: torch.Tensor, active_diseases: list[str], severity: torch.Tensor) -> torch.Tensor:
        residual = self.texture_branch(image, disease_embed)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.35, 0.35)
        boosted = torch.clamp(image + 0.35 * residual, 0.0, 1.0)
        boosted = apply_disease_heuristic_boost(boosted, active_diseases, severity)
        return clamp_and_normalize_image(boosted)

    def forward(
        self,
        image: torch.Tensor,
        prompt: list[str],
        disease_embed: torch.Tensor,
        dense_flow: torch.Tensor,
        landmark_heatmap: torch.Tensor,
        texture_residual: torch.Tensor,
        severity: torch.Tensor,
        active_diseases: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        disease_embed = self._stabilize_condition(disease_embed)
        control_image = self.control_adapter(dense_flow, landmark_heatmap, texture_residual, severity)
        if self.pipeline is None:
            return self._fallback_render(image, disease_embed, active_diseases, severity), control_image

        self.set_lora_scales(active_diseases, severity)
        for processor in self.attn_processors:
            processor.set_condition(disease_embed)
        try:
            output = self.pipeline(
                prompt=prompt,
                image=image,
                control_image=control_image,
                strength=self.config.strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                controlnet_conditioning_scale=self.config.controlnet_scale,
                output_type="pt",
                generator=torch.Generator(device=self.config.device).manual_seed(42),
            )
            rendered = torch.nan_to_num(output.images, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            rendered = clamp_and_normalize_image(rendered)
            if not is_render_tensor_valid(rendered):
                return self._fallback_render(image, disease_embed, active_diseases, severity), control_image
            return rendered, control_image
        except Exception:
            return self._fallback_render(image, disease_embed, active_diseases, severity), control_image
        finally:
            for processor in self.attn_processors:
                processor.set_condition(None)


class DiseaseAwareDiffusionPipeline(nn.Module):
    def __init__(self, warp_layer: DenseWarpLayer, texture_branch: TextureBranch, renderer: DiseaseDiffusionRenderer) -> None:
        super().__init__()
        self.warp_layer = warp_layer
        self.texture_branch = texture_branch
        self.renderer = renderer
        if self.renderer.texture_branch is not self.texture_branch:
            raise ValueError("renderer.texture_branch must share the same module instance.")

    def forward(
        self,
        image: torch.Tensor,
        landmarks: torch.Tensor,
        delta_landmarks: torch.Tensor,
        disease_embed: torch.Tensor,
        prompt: list[str],
        severity: torch.Tensor,
        active_diseases: list[str],
    ) -> dict[str, torch.Tensor]:
        assert image.ndim == 4
        assert landmarks.shape == delta_landmarks.shape
        target_landmarks = landmarks + torch.nan_to_num(delta_landmarks, nan=0.0, posinf=0.0, neginf=0.0).clamp(-32.0, 32.0)
        warped_image, dense_flow = self.warp_layer(clamp_and_normalize_image(image), landmarks, target_landmarks)
        texture_residual = self.texture_branch(warped_image, disease_embed)
        texture_residual = torch.nan_to_num(texture_residual, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.35, 0.35)
        textured_image = torch.clamp(warped_image + texture_residual, 0.0, 1.0)
        landmark_heatmap = build_landmark_heatmap(landmarks, target_landmarks, textured_image.shape[-2:])
        rendered_image, control_image = self.renderer(
            image=textured_image,
            prompt=prompt,
            disease_embed=disease_embed,
            dense_flow=dense_flow,
            landmark_heatmap=landmark_heatmap,
            texture_residual=texture_residual,
            severity=severity,
            active_diseases=active_diseases,
        )
        return {
            "warped_image": warped_image,
            "dense_flow": dense_flow,
            "texture_residual": texture_residual,
            "textured_image": textured_image,
            "landmark_heatmap": landmark_heatmap,
            "control_image": control_image,
            "rendered_image": rendered_image,
            "target_landmarks": target_landmarks,
        }
