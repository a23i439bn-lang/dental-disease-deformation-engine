"""研究用の総合生成スクリプト。

参照画像、口領域マスク、ランドマーク変形、最終画像生成までを含む
大きな実験パイプライン。今の軽量検証では必要部品の参照元として使う。
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

from utils.disease_priors import resolve_preset_disease_name


DEFAULT_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
DEFAULT_CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
DEFAULT_NEGATIVE_PROMPT = (
    "anime, cartoon, illustration, painting, cgi, 3d render, fake skin, plastic skin, doll face, "
    "airbrushed, stylized, unreal teeth, extra teeth, malformed mouth, duplicated lips, warped jaw, "
    "deformed face, oversaturated, blurry, low quality"
)
SEVERITY_BY_STRENGTH = {"mild": 0.45, "moderate": 0.78, "severe": 1.0, "baseline": 0.0}
LOCAL_ENHANCEMENT_BY_STRENGTH = {"mild": 0.46, "moderate": 0.72, "severe": 0.95}
MOUTH_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]
ANCHOR_INDICES = [1, 4, 6, 8, 9, 10, 33, 61, 93, 127, 152, 172, 197, 234, 263, 291, 323, 356, 389, 454]
INNER_MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
UPPER_GUM_HINT_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LOWER_GUM_HINT_INDICES = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
DISEASE_SETTINGS: dict[str, dict[str, Any]] = {
    "虫歯": {
        "focus_mask": "teeth",
        "prompt_terms": [
            "{strength} dental caries",
            "visible tooth decay on teeth only",
            "do not change chin jaw cheeks",
        ],
        "color_shift": np.array([18.0, -10.0, -26.0], dtype=np.float32),
        "blend_alpha": 0.72,
    },
    "出っ歯": {
        "focus_mask": "teeth",
        "prompt_terms": [
            "{strength} maxillary protrusion",
            "forward protruding upper teeth only",
            "do not change chin jaw cheeks",
        ],
        "bias": np.array([0.015, 0.0], dtype=np.float32),
        "reference_x_scale": 0.038,
        "heuristic_x_scale": 0.046,
    },
    "すきっ歯": {
        "focus_mask": "teeth",
        "prompt_terms": [
            "{strength} diastema",
            "visible spacing between front teeth only",
            "do not change chin jaw cheeks",
        ],
        "bias": np.array([0.01, 0.0], dtype=np.float32),
        "reference_x_scale": 0.028,
        "heuristic_x_scale": 0.036,
    },
    "normal": {
        "focus_mask": "composite",
        "prompt_terms": ["healthy mouth"],
        "bias": np.array([0.0, 0.0], dtype=np.float32),
    },
}

''' モジュールのパラメータを収集する関数 '''
@dataclass
class MultiRegionMask:
    mouth: Image.Image
    teeth: Image.Image
    gums: Image.Image
    composite: Image.Image
    preview: Image.Image

''' 研究用モジュールを構築する関数 '''
@dataclass
class DiseasePreset:
    name: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    strength_label: str
    denoise_strength: float

''' 疾患名を正規化する関数 '''
@dataclass
class TrainingExportPackage:
    manifest: dict[str, Any]
    images: dict[str, Image.Image]

''' 研究用モジュールを構築する関数 '''
@dataclass
class DenseFlowWarpResult:
    warped_image: Image.Image
    heatmap: Image.Image

''' モジュールのパラメータを収集する関数 '''
@dataclass
class IdentityMetrics:
    cosine_similarity: float
    identity_loss: float
    backbone_name: str
    used_fallback: bool


class FlowRefinementCNN(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleIdentityEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image).flatten(1)
        return F.normalize(self.proj(features), dim=-1)


class ArcFaceIdentityEncoder(nn.Module):
    def __init__(self, device: str, embedding_dim: int = 512) -> None:
        super().__init__()
        self.device_name = device
        self.fallback = SimpleIdentityEncoder(embedding_dim=embedding_dim)
        self.fallback_name = "simple_cnn_fallback"
        self._insightface_model = None
        self._backbone_name = self.fallback_name
        self._used_fallback = True
        self._try_load_insightface()

    def _try_load_insightface(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            providers = ["CPUExecutionProvider"]
            if self.device_name == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self.device_name == "cuda" else -1, det_size=(320, 320))
            self._insightface_model = app
            self._backbone_name = "insightface_arcface_buffalo_l"
            self._used_fallback = False
        except Exception:
            self._insightface_model = None

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    def _forward_insightface(self, image: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare(image)
        samples = prepared.detach().cpu().permute(0, 2, 3, 1).numpy()
        embeddings = []
        for sample in samples:
            bgr = cv2.cvtColor((sample * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            faces = self._insightface_model.get(bgr)
            if faces:
                embedding = torch.from_numpy(faces[0].normed_embedding).float()
            else:
                fallback_input = torch.from_numpy(sample).permute(2, 0, 1).unsqueeze(0).float()
                embedding = self.fallback(fallback_input).squeeze(0).cpu()
            embeddings.append(embedding)
        return F.normalize(torch.stack(embeddings, dim=0).to(image.device), dim=-1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, str, bool]:
        if self._insightface_model is not None:
            return self._forward_insightface(image), self._backbone_name, self._used_fallback
        return self.fallback(self._prepare(image)), self._backbone_name, self._used_fallback


class DenseFlowWarpingModule:
    def __init__(self) -> None:
        self.refinement_cnn = FlowRefinementCNN()

    @staticmethod
    def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        image = image_tensor.detach().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
        return Image.fromarray((image * 255.0).astype(np.uint8))

    @staticmethod
    def _build_base_grid(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xs, ys], dim=-1).unsqueeze(0)

    def generate_dense_flow(
        self,
        source_landmarks: np.ndarray,
        target_landmarks: np.ndarray,
        image_size: tuple[int, int],
        device: str,
        sigma: float,
    ) -> torch.Tensor:
        width, height = image_size
        src = torch.from_numpy(source_landmarks.astype(np.float32)).to(device)
        dst = torch.from_numpy(target_landmarks.astype(np.float32)).to(device)
        disp = dst - src
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1)
        delta = grid[:, :, None, :] - src[None, None, :, :]
        dist2 = (delta * delta).sum(dim=-1)
        weights = torch.exp(-dist2 / (2.0 * max(sigma, 1.0) ** 2))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        flow = (weights[..., None] * disp[None, None, :, :]).sum(dim=2)
        return flow.permute(2, 0, 1).unsqueeze(0)

    def build_roi_mask(self, image_size: tuple[int, int], roi_box: tuple[int, int, int, int], device: str) -> torch.Tensor:
        width, height = image_size
        left, top, right, bottom = roi_box
        mask = torch.zeros(1, 1, height, width, device=device, dtype=torch.float32)
        mask[:, :, top:bottom, left:right] = 1.0
        return F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7).clamp(0.0, 1.0)

    def refine_flow(self, sparse_flow: torch.Tensor, roi_mask: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        refinement_input = torch.cat([sparse_flow, roi_mask, image_tensor.mean(dim=1, keepdim=True)], dim=1)
        residual = self.refinement_cnn(refinement_input)
        return sparse_flow + residual * roi_mask

    def warp(self, image: Image.Image, dense_flow: torch.Tensor, device: str) -> torch.Tensor:
        image_tensor = self._pil_to_tensor(image, device)
        _, _, height, width = image_tensor.shape
        norm_flow = torch.zeros_like(dense_flow)
        norm_flow[:, 0] = dense_flow[:, 0] / max(width - 1, 1) * 2.0
        norm_flow[:, 1] = dense_flow[:, 1] / max(height - 1, 1) * 2.0
        base_grid = self._build_base_grid(height, width, device, image_tensor.dtype)
        sampling_grid = base_grid - norm_flow.permute(0, 2, 3, 1)
        return F.grid_sample(image_tensor, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)

    @staticmethod
    def visualize_flow(flow_tensor: torch.Tensor) -> Image.Image:
        flow_np = flow_tensor.detach().cpu()[0].permute(1, 2, 0).numpy()
        magnitude = np.linalg.norm(flow_np, axis=-1)
        scaled = np.clip(magnitude / max(float(magnitude.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))

    def warp_from_landmarks(
        self,
        image: Image.Image,
        source_landmarks: np.ndarray,
        target_landmarks: np.ndarray,
        roi_box: tuple[int, int, int, int],
        device: str,
        sigma: float,
    ) -> DenseFlowWarpResult:
        image_tensor = self._pil_to_tensor(image, device)
        sparse_flow = self.generate_dense_flow(source_landmarks, target_landmarks, image.size, device, sigma)
        roi_mask = self.build_roi_mask(image.size, roi_box, device)
        dense_flow = self.refine_flow(sparse_flow, roi_mask, image_tensor)
        warped_tensor = self.warp(image, dense_flow, device)
        return DenseFlowWarpResult(
            warped_image=self._tensor_to_pil(warped_tensor),
            heatmap=self.visualize_flow(dense_flow),
        )


def get_disease_setting(preset_name: str) -> dict[str, Any]:
    if preset_name == "normal":
        return DISEASE_SETTINGS["normal"]
    resolved_name = resolve_preset_disease_name(preset_name, DISEASE_SETTINGS.keys())
    return DISEASE_SETTINGS.get(resolved_name, DISEASE_SETTINGS["normal"])


def resolve_strength_value(strength_label: str, override: float | None = None) -> float:
    if override is not None:
        return float(np.clip(override, 0.0, 1.0))
    return SEVERITY_BY_STRENGTH.get(strength_label, 0.5)


def apply_disease_landmark_prior(
    target: np.ndarray,
    preset_name: str,
    scale: np.ndarray,
    severity: float,
    gums_mean: float,
    mode: str,
) -> np.ndarray:
    preset_name = resolve_preset_disease_name(preset_name, DISEASE_SETTINGS.keys())
    settings = get_disease_setting(preset_name)
    bias = settings.get("bias", np.array([0.0, 0.0], dtype=np.float32))
    if preset_name in {"出っ歯", "すきっ歯"}:
        x_scale = settings["reference_x_scale"] if mode == "reference" else settings["heuristic_x_scale"]
        if mode == "reference":
            x_offsets = np.linspace(-1.0, 1.0, len(target), dtype=np.float32)[:, None]
        else:
            x_offsets = np.sin(np.linspace(0, np.pi * 2, len(target), dtype=np.float32))[:, None]
        return target + np.concatenate([x_offsets, np.zeros_like(x_offsets)], axis=1) * scale[None, :] * x_scale * severity
    return target


def load_image(image_path: Path, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    new_w = max(8, (resized.width // 8) * 8)
    new_h = max(8, (resized.height // 8) * 8)
    return resized.resize((new_w, new_h), Image.LANCZOS)


def build_canny_image(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    np_image = np.array(image)
    edges = cv2.Canny(np_image, low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges] * 3, axis=2))


def polygon_mask(image_size: tuple[int, int], points: np.ndarray, blur_sigma: float = 0.0) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return mask


def load_presets(preset_path: Path) -> dict[str, Any]:
    with preset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_preset(mode: str, disease_name: str, strength: str, preset_data: dict[str, Any]) -> DiseasePreset:
    common = preset_data["common"]
    diseases = preset_data["diseases"]
    if mode == "normal":
        return DiseasePreset(
            name="normal",
            prompt=common["baseline_prompt"],
            negative_prompt=common.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            guidance_scale=common.get("baseline_guidance_scale", 4.5),
            num_inference_steps=common.get("baseline_steps", 24),
            strength_label="baseline",
            denoise_strength=common.get("baseline_denoise_strength", 0.12),
        )
    try:
        resolved_name = resolve_preset_disease_name(disease_name, diseases.keys())
    except KeyError as exc:
        available = ", ".join(sorted(diseases.keys()))
        raise KeyError(f"Unknown disease preset '{disease_name}'. Available: {available}") from exc
    disease = diseases[resolved_name]
    prompt_map = disease.get("prompts", {})
    if strength not in prompt_map:
        available = ", ".join(sorted(prompt_map.keys()))
        raise KeyError(f"Unknown strength '{strength}' for disease '{resolved_name}'. Available: {available}")
    return DiseasePreset(
        name=resolved_name,
        prompt=prompt_map[strength],
        negative_prompt=disease.get("negative_prompt", common.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
        guidance_scale=disease.get("guidance_scale", common.get("guidance_scale", 6.5)),
        num_inference_steps=disease.get("steps", common.get("steps", 30)),
        strength_label=strength,
        denoise_strength=disease.get("denoise_strength", common.get("denoise_strength", 0.18)),
    )


def resolve_device(requested_device: str) -> tuple[str, torch.dtype]:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return "cuda", torch.float16
    if requested_device == "cpu":
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def set_seed(seed: int | None) -> int:
    final_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)
    return final_seed


def build_pipeline(model_id: str, controlnet_id: str, device: str, dtype: torch.dtype, lora_path: str | None, lora_scale: float):
    try:
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype, local_files_only=True)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    except Exception:
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def detect_face_landmarks(image: Image.Image) -> np.ndarray:
    width, height = image.size
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        try:
            result = face_mesh.process(np.array(image))
        finally:
            face_mesh.close()
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)
    return detect_face_landmarks_with_heuristic(image)


def detect_face_landmarks_with_heuristic(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces: list[Any] | tuple[Any, ...] = []
    if os.environ.get("ENABLE_HAAR_FALLBACK", "0") == "1":
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    width, height = image.size
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    else:
        x = int(width * 0.18)
        y = int(height * 0.12)
        w = int(width * 0.64)
        h = int(height * 0.76)

    landmarks = np.zeros((468, 2), dtype=np.float32)
    for idx in ANCHOR_INDICES:
        landmarks[idx] = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    anchor_layout = {
        1: (0.50, 0.58), 4: (0.50, 0.67), 6: (0.50, 0.42), 8: (0.50, 0.78), 9: (0.50, 0.20),
        10: (0.50, 0.12), 33: (0.28, 0.35), 61: (0.36, 0.66), 93: (0.20, 0.52), 127: (0.14, 0.36),
        152: (0.50, 0.92), 172: (0.36, 0.82), 197: (0.50, 0.52), 234: (0.08, 0.52), 263: (0.72, 0.35),
        291: (0.64, 0.66), 323: (0.80, 0.52), 356: (0.86, 0.36), 389: (0.92, 0.52), 454: (0.95, 0.52),
    }
    for idx, (rx, ry) in anchor_layout.items():
        landmarks[idx] = np.array([x + w * rx, y + h * ry], dtype=np.float32)

    mouth_center = estimate_mouth_center_from_face_roi(rgb, x, y, w, h)
    mouth_radius = np.array([w * 0.15, h * 0.045], dtype=np.float32)
    for offset, idx in enumerate(MOUTH_INDICES):
        angle = (2.0 * np.pi * offset) / len(MOUTH_INDICES)
        landmarks[idx] = mouth_center + np.array([np.cos(angle) * mouth_radius[0], np.sin(angle) * mouth_radius[1]], dtype=np.float32)

    inner_radius = mouth_radius * np.array([0.72, 0.52], dtype=np.float32)
    for offset, idx in enumerate(INNER_MOUTH_INDICES):
        angle = (2.0 * np.pi * offset) / len(INNER_MOUTH_INDICES)
        landmarks[idx] = mouth_center + np.array([np.cos(angle) * inner_radius[0], np.sin(angle) * inner_radius[1]], dtype=np.float32)
    return landmarks


def estimate_mouth_center_from_face_roi(rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    left = max(0, int(x + w * 0.18))
    right = min(width, int(x + w * 0.82))
    top = max(0, int(y + h * 0.48))
    bottom = min(height, int(y + h * 0.74))
    roi = rgb[top:bottom, left:right]
    if roi.size == 0:
        return np.array([x + w * 0.5, y + h * 0.60], dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    red = rgb[top:bottom, left:right, 0].astype(np.float32) / 255.0
    green = rgb[top:bottom, left:right, 1].astype(np.float32) / 255.0
    a_channel = lab[:, :, 1].astype(np.float32)
    lip_score = np.clip((red - green) * 1.8 + sat * 0.8 + (a_channel - 128.0) / 64.0, 0.0, 1.0)
    teeth_score = np.clip(value * (1.0 - sat), 0.0, 1.0)
    row_weight = np.linspace(1.1, 0.85, roi.shape[0], dtype=np.float32)[:, None]
    center_weight = 1.0 - np.abs(np.linspace(-1.0, 1.0, roi.shape[1], dtype=np.float32))[None, :] * 0.35
    score = (lip_score * 0.8 + teeth_score * 0.6) * row_weight * center_weight
    threshold = np.percentile(score, 88)
    ys, xs = np.where(score >= threshold)
    if len(xs) == 0:
        return np.array([x + w * 0.5, y + h * 0.60], dtype=np.float32)
    weights = score[ys, xs] + 1e-6
    cx = left + float(np.average(xs, weights=weights))
    cy = top + float(np.average(ys, weights=weights))
    cy = float(np.clip(cy, y + h * 0.52, y + h * 0.66))
    return np.array([cx, cy], dtype=np.float32)


def draw_landmarks(image: Image.Image, landmarks: np.ndarray, indices: list[int], color: tuple[int, int, int]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for idx in indices:
        x, y = landmarks[idx]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return canvas


def compute_reference_guided_targets(source_landmarks: np.ndarray, reference_landmarks: np.ndarray, reference_mix: float) -> np.ndarray:
    src_pts = source_landmarks[MOUTH_INDICES]
    ref_pts = reference_landmarks[MOUTH_INDICES]
    src_min = src_pts.min(axis=0)
    src_max = src_pts.max(axis=0)
    ref_min = ref_pts.min(axis=0)
    ref_max = ref_pts.max(axis=0)
    src_size = np.maximum(src_max - src_min, 1.0)
    ref_size = np.maximum(ref_max - ref_min, 1.0)
    ref_norm = (ref_pts - ref_min) / ref_size
    transferred = src_min + ref_norm * src_size
    return src_pts + reference_mix * (transferred - src_pts)


def build_mouth_roi_box(landmarks: np.ndarray, image_size: tuple[int, int], padding_ratio: float) -> tuple[int, int, int, int]:
    mouth = landmarks[MOUTH_INDICES]
    min_xy = mouth.min(axis=0)
    max_xy = mouth.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)
    pad = size * padding_ratio
    left = max(0, int(min_xy[0] - pad[0]))
    top = max(0, int(min_xy[1] - pad[1]))
    right = min(image_size[0], int(max_xy[0] + pad[0]))
    bottom = min(image_size[1], int(max_xy[1] + pad[1]))
    return left, top, right, bottom


def warp_image_with_landmark_displacements(
    image: Image.Image,
    source_landmarks: np.ndarray,
    target_mouth_points: np.ndarray,
    roi_box: tuple[int, int, int, int],
    landmark_sigma: float,
    device: str,
) -> tuple[Image.Image, Image.Image]:
    control_src = np.concatenate([source_landmarks[MOUTH_INDICES], source_landmarks[ANCHOR_INDICES]], axis=0)
    control_dst = np.concatenate([target_mouth_points, source_landmarks[ANCHOR_INDICES]], axis=0)
    result = DenseFlowWarpingModule().warp_from_landmarks(image, control_src, control_dst, roi_box, device, landmark_sigma)
    return result.warped_image, result.heatmap


def build_oral_region_masks(image: Image.Image, landmarks: np.ndarray, blur_ratio: float) -> MultiRegionMask:
    width, height = image.size
    rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    mouth_points = landmarks[MOUTH_INDICES]
    inner_points = landmarks[INNER_MOUTH_INDICES]
    upper_hint = landmarks[UPPER_GUM_HINT_INDICES]
    lower_hint = landmarks[LOWER_GUM_HINT_INDICES]
    sigma = max(3.0, min(width, height) * blur_ratio)
    mouth_mask = polygon_mask(image.size, mouth_points, blur_sigma=sigma)
    inner_mask = polygon_mask(image.size, inner_points, blur_sigma=max(1.0, sigma * 0.6))

    value = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)

    inner_pixels = inner_mask > 0
    if not np.any(inner_pixels):
        inner_pixels = mouth_mask > 0
    tooth_v = np.percentile(value[inner_pixels], 72)
    tooth_s = np.percentile(sat[inner_pixels], 45)
    teeth_binary = ((value >= tooth_v) & (sat <= tooth_s) & inner_pixels).astype(np.uint8) * 255
    teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_teeth_area = max(1, int(np.count_nonzero(inner_mask) * 0.18))
    if cv2.countNonZero(teeth_binary) < min_teeth_area:
        tooth_v = np.percentile(value[inner_pixels], 60)
        tooth_s = np.percentile(sat[inner_pixels], 55)
        teeth_binary = ((value >= tooth_v) & (sat <= tooth_s) & inner_pixels).astype(np.uint8) * 255
        teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    teeth_binary = cv2.dilate(teeth_binary, np.ones((7, 7), np.uint8), iterations=1)
    if cv2.countNonZero(teeth_binary) < min_teeth_area:
        fallback_teeth = cv2.bitwise_and(mouth_mask, inner_mask)
        fallback_teeth = cv2.morphologyEx(fallback_teeth, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        teeth_binary = cv2.bitwise_or(teeth_binary, fallback_teeth)

    upper_mask = polygon_mask(image.size, upper_hint)
    lower_mask = polygon_mask(image.size, lower_hint)
    gum_region = cv2.bitwise_and(mouth_mask, cv2.bitwise_not(inner_mask))
    gum_region = cv2.bitwise_or(gum_region, upper_mask)
    gum_region = cv2.bitwise_or(gum_region, lower_mask)
    gum_pixels = gum_region > 0
    if not np.any(gum_pixels):
        gum_pixels = mouth_mask > 0
    gum_a = np.percentile(a_channel[gum_pixels], 58)
    gum_b = np.percentile(b_channel[gum_pixels], 52)
    gums_binary = ((a_channel >= gum_a) & (b_channel >= gum_b) & (gum_region > 0)).astype(np.uint8) * 255
    gums_binary = cv2.morphologyEx(gums_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gums_binary = cv2.morphologyEx(gums_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    composite = np.maximum.reduce([mouth_mask, teeth_binary, gums_binary])
    mouth_soft = cv2.GaussianBlur(mouth_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    teeth_soft = cv2.GaussianBlur(teeth_binary, (0, 0), sigmaX=max(1.0, sigma * 0.5), sigmaY=max(1.0, sigma * 0.5))
    gums_soft = cv2.GaussianBlur(gums_binary, (0, 0), sigmaX=max(1.0, sigma * 0.5), sigmaY=max(1.0, sigma * 0.5))
    composite_soft = cv2.GaussianBlur(composite, (0, 0), sigmaX=max(1.0, sigma * 0.7), sigmaY=max(1.0, sigma * 0.7))

    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[:, :, 0] = gums_binary
    preview[:, :, 1] = teeth_binary
    preview[:, :, 2] = mouth_mask
    return MultiRegionMask(
        mouth=Image.fromarray(mouth_soft, mode="L"),
        teeth=Image.fromarray(teeth_soft, mode="L"),
        gums=Image.fromarray(gums_soft, mode="L"),
        composite=Image.fromarray(composite_soft, mode="L"),
        preview=Image.fromarray(preview, mode="RGB"),
    )


def build_region_aware_control_image(init_image: Image.Image, canny_image: Image.Image, oral_masks: MultiRegionMask, deformation_heatmap: Image.Image) -> Image.Image:
    canny = np.array(canny_image)
    gums = np.array(oral_masks.gums)
    teeth = np.array(oral_masks.teeth)
    heat = np.array(deformation_heatmap.convert("L"))
    control = canny.copy()
    control[:, :, 0] = np.maximum(control[:, :, 0], gums)
    control[:, :, 1] = np.maximum(control[:, :, 1], teeth)
    control[:, :, 2] = np.maximum(control[:, :, 2], heat)
    return Image.fromarray(control, mode="RGB")


def blend_with_focus_area(source_image: Image.Image, generated_image: Image.Image, focus_mask: Image.Image) -> Image.Image:
    source = np.array(source_image).astype(np.float32)
    generated = np.array(generated_image).astype(np.float32)
    alpha = np.array(focus_mask).astype(np.float32)[:, :, None] / 255.0
    blended = source * (1.0 - alpha) + generated * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def select_focus_mask_for_disease(preset: DiseasePreset, oral_masks: MultiRegionMask) -> Image.Image:
    focus_mask_name = get_disease_setting(preset.name).get("focus_mask", "composite")
    return getattr(oral_masks, focus_mask_name, oral_masks.composite)


def compose_generation_prompt(preset: DiseasePreset, has_reference: bool, prompt_suffix: str) -> str:
    strength_terms = {"mild": "mild", "moderate": "moderate", "severe": "severe", "baseline": "normal"}
    strength = strength_terms.get(preset.strength_label, preset.strength_label)
    identity_terms = [
        "same person",
        "preserve identity",
        "clinical dental photo",
        "photorealistic",
        "natural skin texture",
        "realistic teeth",
        "realistic gums",
    ]
    disease_terms = [term.format(strength=strength) for term in get_disease_setting(preset.name).get("prompt_terms", ["healthy mouth"])]
    control_terms = []
    if has_reference:
        control_terms.append("reference-guided mouth pattern")
    if prompt_suffix:
        control_terms.append(prompt_suffix)
    return ", ".join(part for part in disease_terms + identity_terms + control_terms if part)


def normalize_mouth_landmarks(mouth_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = mouth_points.mean(axis=0)
    scale = np.maximum(mouth_points.max(axis=0) - mouth_points.min(axis=0), 1.0)
    normalized = (mouth_points - center) / scale
    return normalized.astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def infer_target_mouth_points(
    source_landmarks: np.ndarray,
    preset: DiseasePreset,
    oral_masks: MultiRegionMask,
    reference_landmarks: np.ndarray | None,
    reference_mix: float,
    severity_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_mouth_points = source_landmarks[MOUTH_INDICES]
    _, _, scale = normalize_mouth_landmarks(source_mouth_points)
    gums_mean = float(np.array(oral_masks.gums).mean() / 255.0)
    severity = resolve_strength_value(preset.strength_label, severity_override)
    if reference_landmarks is not None:
        target = compute_reference_guided_targets(source_landmarks, reference_landmarks, reference_mix)
        target = apply_disease_landmark_prior(target, preset.name, scale, severity, gums_mean, mode="reference")
        policy_info = {"policy_mode": "reference_heuristic", "vision_conditioning": False, "text_conditioning": False, "severity": severity}
    else:
        target = apply_disease_landmark_prior(source_mouth_points.copy(), preset.name, scale, severity, gums_mean, mode="heuristic")
        policy_info = {"policy_mode": "heuristic", "vision_conditioning": False, "text_conditioning": False, "severity": severity}
    return target.astype(np.float32), policy_info


def extract_roi(image: Image.Image, roi_box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(roi_box)


def create_disease_diff_outputs(source_image: Image.Image, result_image: Image.Image, roi_box: tuple[int, int, int, int], diff_threshold: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    src = np.array(source_image)
    dst = np.array(result_image)
    diff = cv2.absdiff(src, dst)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    left, top, right, bottom = roi_box
    roi_mask = np.zeros_like(diff_gray, dtype=np.uint8)
    roi_mask[top:bottom, left:right] = 255
    _, binary = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, roi_mask)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    heat = cv2.applyColorMap(np.clip(diff_gray * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.bitwise_and(heat, heat, mask=roi_mask)
    extracted = np.zeros_like(dst)
    extracted[binary > 0] = dst[binary > 0]
    return Image.fromarray(binary, mode="L"), Image.fromarray(heat), Image.fromarray(extracted)


@torch.no_grad()
def compute_identity_metrics(source_image: torch.Tensor, edited_image: torch.Tensor, device: str) -> IdentityMetrics:
    model = ArcFaceIdentityEncoder(device=device)
    src_embeddings, backbone_name, used_fallback = model(source_image.to(device))
    dst_embeddings, _, _ = model(edited_image.to(device))
    cosine = F.cosine_similarity(src_embeddings, dst_embeddings, dim=-1).mean()
    identity_loss = 1.0 - cosine
    return IdentityMetrics(
        cosine_similarity=float(cosine.detach().cpu()),
        identity_loss=float(identity_loss.detach().cpu()),
        backbone_name=backbone_name,
        used_fallback=used_fallback,
    )


def apply_disease_local_enhancement(image: Image.Image, preset: DiseasePreset, oral_masks: MultiRegionMask) -> Image.Image:
    settings = get_disease_setting(preset.name)
    if "color_shift" not in settings:
        return image
    strength = LOCAL_ENHANCEMENT_BY_STRENGTH.get(preset.strength_label, 0.45)
    rgb = np.array(image).astype(np.float32)
    gum_alpha = np.array(oral_masks.gums).astype(np.float32)[:, :, None] / 255.0
    teeth_alpha = np.array(oral_masks.teeth).astype(np.float32)[:, :, None] / 255.0
    if preset.name == "虫歯":
        tinted = np.clip(rgb - np.abs(settings["color_shift"][None, None, :]) * strength, 0, 255)
        rgb = rgb * (1.0 - teeth_alpha * settings["blend_alpha"]) + tinted * (teeth_alpha * settings["blend_alpha"])
    elif preset.name == "出っ歯":
        return image
    elif preset.name == "すきっ歯":
        return image
    elif preset.name == "gingivitis":
        tinted = np.clip(rgb + settings["color_shift"][None, None, :] * strength, 0, 255)
        rgb = rgb * (1.0 - gum_alpha * settings["blend_alpha"]) + tinted * (gum_alpha * settings["blend_alpha"])
    elif preset.name == "oral_cancer":
        darkened = np.clip(
            rgb * (1.0 - settings["darken_scale"] * strength) + settings["color_shift"][None, None, :] * strength,
            0,
            255,
        )
        rgb = rgb * (1.0 - gum_alpha * settings["blend_alpha"]) + darkened * (gum_alpha * settings["blend_alpha"])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def save_outputs(output_dir: Path, images: dict[str, Image.Image], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image_path = output_dir / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def build_training_export_package(
    source_image: Image.Image,
    init_image: Image.Image,
    final_generated: Image.Image,
    source_landmarks: np.ndarray,
    target_mouth_points: np.ndarray,
    roi_box: tuple[int, int, int, int],
    oral_masks: MultiRegionMask,
    preset: DiseasePreset,
    policy_info: dict[str, Any],
) -> TrainingExportPackage:
    target_landmarks = source_landmarks.copy()
    target_landmarks[MOUTH_INDICES] = target_mouth_points
    roi_left, roi_top, roi_right, roi_bottom = roi_box
    manifest = {
        "version": 1,
        "task": "medical_facial_deformation_training_sample",
        "disease": preset.name,
        "strength": preset.strength_label,
        "severity": policy_info.get("severity"),
        "policy_mode": policy_info.get("policy_mode"),
        "roi_box": {"left": roi_left, "top": roi_top, "right": roi_right, "bottom": roi_bottom},
        "source_landmarks_468": source_landmarks.tolist(),
        "target_landmarks_468": target_landmarks.tolist(),
        "mouth_indices": MOUTH_INDICES,
        "anchor_indices": ANCHOR_INDICES,
        "files": {
            "source_image": "training/source_image.png",
            "init_image": "training/init_image.png",
            "generated_image": "training/generated_image.png",
            "mouth_mask": "training/mouth_mask.png",
            "teeth_mask": "training/teeth_mask.png",
            "gums_mask": "training/gums_mask.png",
            "composite_mask": "training/composite_mask.png",
        },
    }
    images = {
        "training/source_image.png": source_image,
        "training/init_image.png": init_image,
        "training/generated_image.png": final_generated,
        "training/mouth_mask.png": oral_masks.mouth,
        "training/teeth_mask.png": oral_masks.teeth,
        "training/gums_mask.png": oral_masks.gums,
        "training/composite_mask.png": oral_masks.composite,
    }
    return TrainingExportPackage(manifest=manifest, images=images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file dental image generator.")
    parser.add_argument("--input", required=True, help="Path to the base face image.")
    parser.add_argument("--disease-reference", help="Optional reference image for mouth shape guidance.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory to save generated files.")
    parser.add_argument("--mode", choices=["normal", "disease"], default="disease")
    parser.add_argument("--disease", default="虫歯", help="Disease preset name.")
    parser.add_argument("--strength", default="moderate", help="Symptom strength preset.")
    parser.add_argument("--preset-file", default="presets/disease_prompts.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--controlnet-id", default=DEFAULT_CONTROLNET_ID)
    parser.add_argument("--lora-path", default=None, help="Optional LoRA weights.")
    parser.add_argument("--lora-scale", type=float, default=0.9)
    parser.add_argument("--size", type=int, default=768, help="Target size for the shorter image side.")
    parser.add_argument("--controlnet-scale", type=float, default=0.7)
    parser.add_argument("--reference-mix", type=float, default=0.75)
    parser.add_argument("--landmark-sigma", type=float, default=28.0)
    parser.add_argument("--severity", type=float, default=None, help="Optional severity override in [0, 1].")
    parser.add_argument("--roi-padding-ratio", type=float, default=0.6)
    parser.add_argument("--focus-blur-ratio", type=float, default=0.02)
    parser.add_argument("--export-training-package", action="store_true")
    parser.add_argument("--canny-low-threshold", type=int, default=70)
    parser.add_argument("--canny-high-threshold", type=int, default=170)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--denoise-strength", type=float, default=None)
    parser.add_argument("--diff-threshold", type=int, default=18)
    parser.add_argument("--skip-reference-warp", action="store_true")
    parser.add_argument("--prompt-suffix", default="", help="Extra phrase appended to the selected prompt.")
    parser.add_argument("--compute-identity-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    preset_path = Path(args.preset_file)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")
    if args.disease_reference and not Path(args.disease_reference).exists():
        raise FileNotFoundError(f"Disease reference image not found: {args.disease_reference}")
    if args.lora_path and not Path(args.lora_path).exists():
        raise FileNotFoundError(f"LoRA weights not found: {args.lora_path}")

    preset_data = load_presets(preset_path)
    preset = build_preset(args.mode, args.disease, args.strength, preset_data)
    source_image = load_image(input_path, args.size)
    source_landmarks = detect_face_landmarks(source_image)
    reference_image = None
    reference_landmarks = None
    if args.disease_reference:
        reference_image = load_image(Path(args.disease_reference), args.size)
        reference_landmarks = detect_face_landmarks(reference_image)

    init_image = source_image
    deformation_heatmap = Image.new("RGB", source_image.size, (0, 0, 0))
    roi_box = build_mouth_roi_box(source_landmarks, source_image.size, args.roi_padding_ratio)
    device, dtype = resolve_device(args.device)
    seed = set_seed(args.seed)
    prompt = compose_generation_prompt(preset, reference_image is not None, args.prompt_suffix)
    denoise_strength = args.denoise_strength if args.denoise_strength is not None else preset.denoise_strength
    oral_masks = build_oral_region_masks(source_image, source_landmarks, args.focus_blur_ratio)

    use_reference_guidance = reference_landmarks is not None and not args.skip_reference_warp
    target_mouth_points, policy_info = infer_target_mouth_points(
        source_landmarks=source_landmarks,
        preset=preset,
        oral_masks=oral_masks,
        reference_landmarks=reference_landmarks if use_reference_guidance else None,
        reference_mix=args.reference_mix,
        severity_override=args.severity,
    )
    if use_reference_guidance or policy_info["policy_mode"] != "heuristic":
        init_image, deformation_heatmap = warp_image_with_landmark_displacements(
            source_image,
            source_landmarks,
            target_mouth_points,
            roi_box,
            args.landmark_sigma,
            device,
        )

    canny_image = build_canny_image(init_image, args.canny_low_threshold, args.canny_high_threshold)
    control_image = build_region_aware_control_image(init_image, canny_image, oral_masks, deformation_heatmap)
    mouth_focus_mask = select_focus_mask_for_disease(preset, oral_masks)
    if device == "cpu":
        print("CUDA is not available. Running on CPU, so generation may take a long time.")

    pipe = build_pipeline(args.model_id, args.controlnet_id, device, dtype, args.lora_path, args.lora_scale)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=preset.negative_prompt,
        image=init_image,
        control_image=control_image,
        strength=denoise_strength,
        num_inference_steps=preset.num_inference_steps,
        guidance_scale=preset.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    )
    raw_generated = result.images[0]
    final_generated = blend_with_focus_area(source_image, raw_generated, mouth_focus_mask)
    final_generated = apply_disease_local_enhancement(final_generated, preset, oral_masks)

    identity_metrics = None
    if args.compute_identity_metrics:
        src_tensor = torch.from_numpy(np.array(source_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        dst_tensor = torch.from_numpy(np.array(final_generated).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        identity_metrics = compute_identity_metrics(src_tensor, dst_tensor, device)

    mouth_roi_input = extract_roi(source_image, roi_box)
    mouth_roi_init = extract_roi(init_image, roi_box)
    mouth_roi_output = extract_roi(final_generated, roi_box)
    disease_mask, disease_heatmap, disease_extracted = create_disease_diff_outputs(source_image, final_generated, roi_box, args.diff_threshold)
    images = {
        "input_face.png": source_image,
        "init_warped_face.png": init_image,
        "control_canny.png": canny_image,
        "control_oral_regions.png": control_image,
        "raw_generated.png": raw_generated,
        "generated.png": final_generated,
        "mouth_focus_mask.png": mouth_focus_mask,
        "oral_mouth_mask.png": oral_masks.mouth,
        "oral_teeth_mask.png": oral_masks.teeth,
        "oral_gum_mask.png": oral_masks.gums,
        "oral_region_preview.png": oral_masks.preview,
        "mouth_roi_input.png": mouth_roi_input,
        "mouth_roi_init.png": mouth_roi_init,
        "mouth_roi_output.png": mouth_roi_output,
        "disease_mask.png": disease_mask,
        "disease_heatmap.png": disease_heatmap,
        "disease_extracted.png": disease_extracted,
        "deformation_heatmap.png": deformation_heatmap,
        "source_landmarks.png": draw_landmarks(source_image, source_landmarks, MOUTH_INDICES, (255, 64, 64)),
    }
    if reference_image is not None and reference_landmarks is not None:
        images["disease_reference.png"] = reference_image
        images["reference_landmarks.png"] = draw_landmarks(reference_image, reference_landmarks, MOUTH_INDICES, (64, 255, 64))

    metadata = {
        "mode": args.mode,
        "disease": preset.name,
        "strength": preset.strength_label,
        "seed": seed,
        "device": device,
        "model_id": args.model_id,
        "controlnet_id": args.controlnet_id,
        "guidance_scale": preset.guidance_scale,
        "num_inference_steps": preset.num_inference_steps,
        "controlnet_conditioning_scale": args.controlnet_scale,
        "denoise_strength": denoise_strength,
        "lora_path": args.lora_path,
        "lora_scale": args.lora_scale,
        "reference_mix": args.reference_mix,
        "landmark_sigma": args.landmark_sigma,
        "roi_padding_ratio": args.roi_padding_ratio,
        "severity_override": args.severity,
        "diff_threshold": args.diff_threshold,
        "skip_reference_warp": args.skip_reference_warp,
        "has_disease_reference": reference_image is not None,
        "policy_info": policy_info,
        "canny_low_threshold": args.canny_low_threshold,
        "canny_high_threshold": args.canny_high_threshold,
        "mouth_roi_box": {"left": roi_box[0], "top": roi_box[1], "right": roi_box[2], "bottom": roi_box[3]},
        "prompt": prompt,
        "negative_prompt": preset.negative_prompt,
    }
    if identity_metrics is not None:
        metadata["identity_metrics"] = {
            "cosine_similarity": identity_metrics.cosine_similarity,
            "identity_loss": identity_metrics.identity_loss,
            "backbone_name": identity_metrics.backbone_name,
            "used_fallback": identity_metrics.used_fallback,
        }

    training_package = None
    if args.export_training_package:
        training_package = build_training_export_package(
            source_image=source_image,
            init_image=init_image,
            final_generated=final_generated,
            source_landmarks=source_landmarks,
            target_mouth_points=target_mouth_points,
            roi_box=roi_box,
            oral_masks=oral_masks,
            preset=preset,
            policy_info=policy_info,
        )
        metadata["training_package"] = {
            "normalized_mouth_landmarks": normalize_mouth_landmarks(source_landmarks[MOUTH_INDICES])[0].tolist(),
            "target_mouth_landmarks": target_mouth_points.tolist(),
            "manifest_path": "training/manifest.json",
        }
        images.update(training_package.images)

    save_outputs(output_dir, images, metadata)
    if training_package is not None:
        training_dir = output_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        with (training_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(training_package.manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
