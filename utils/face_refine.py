from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import torch
from PIL import Image

from utils.face_regions import (
    CHIN_IDX,
    LEFT_CHIN_SIDE_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MOUTH_CORNER_IDX,
    MOUTH_OUTER_IDX,
    RIGHT_CHIN_SIDE_IDX,
    RIGHT_LOWER_JAW_IDX,
)

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception:
    StableDiffusionInpaintPipeline = None  # type: ignore[assignment]


DEFAULT_SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_SD_PROMPT = (
    "realistic clinical facial photograph, preserve the same person, preserve identity, "
    "natural skin texture, medically plausible orthodontic facial appearance, high detail"
)
DEFAULT_SD_NEGATIVE_PROMPT = (
    "different person, changed identity, distorted face, extra facial features, cartoon, painting, "
    "illustration, cgi, 3d render, plastic skin, over-smoothed skin, malformed jaw"
)


def build_renderer_condition_image(
    original_image: np.ndarray,
    geometric_image: np.ndarray,
    disease_mask: np.ndarray,
) -> np.ndarray:
    if cv2.countNonZero(disease_mask) == 0:
        return geometric_image.copy()
    local_mask = cv2.GaussianBlur((disease_mask.astype(np.float32) / 255.0), (0, 0), sigmaX=3.0, sigmaY=3.0)
    local_mask_3 = np.repeat(np.clip(local_mask, 0.0, 1.0)[:, :, None], 3, axis=2)
    blended = original_image.astype(np.float32) * (1.0 - local_mask_3) + geometric_image.astype(np.float32) * local_mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_edit_bbox(mask: np.ndarray, image_shape: tuple[int, int, int], pad_ratio: float = 0.16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, image_shape[1], image_shape[0])
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)
    pad_x = max(24, int(round(width * pad_ratio)))
    pad_y = max(24, int(round(height * pad_ratio)))
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(image_shape[1], x_max + pad_x + 1),
        min(image_shape[0], y_max + pad_y + 1),
    )


def severity_to_prompt_bucket(severity: float) -> str:
    if severity < 0.35:
        return "mild"
    if severity < 0.8:
        return "moderate"
    return "severe"


def build_sd_prompt(disease: str, severity: float, prompt_override: str) -> str:
    if prompt_override.strip():
        return prompt_override.strip()
    disease_prompts = {
        "mandibular_protrusion": "mandibular prognathism, prominent chin and lower jaw, realistic lower-face skeletal morphology",
        "maxillary_protrusion": "maxillary protrusion, convex profile, prominent upper lip and midface dental protrusion",
        "chin_deviation_left": (
            "chin deviation to the left, mandibular asymmetry, menton shifted left, "
            "leftward chin point, asymmetric jaw contour, realistic lower facial imbalance"
        ),
        "chin_deviation_right": (
            "chin deviation to the right, mandibular asymmetry, menton shifted right, "
            "rightward chin point, asymmetric jaw contour, realistic lower facial imbalance"
        ),
        "occlusal_plane_cant": (
            "occlusal plane cant, asymmetric mouth line, slanted commissure height, "
            "subtle lower facial tilt, realistic dentofacial asymmetry"
        ),
        "occlusal_plane_cant_right": (
            "occlusal plane cant, asymmetric mouth line, slanted commissure height, "
            "subtle lower facial tilt, realistic dentofacial asymmetry"
        ),
    }
    bucket = severity_to_prompt_bucket(severity)
    bucket_text = {
        "mild": "subtle medically plausible change",
        "moderate": "clear medically plausible change",
        "severe": "pronounced but realistic medically plausible change",
    }[bucket]
    return f"{DEFAULT_SD_PROMPT}, {disease_prompts.get(disease, 'realistic facial morphology change')}, {bucket_text}"


def _soften_binary_mask(mask: np.ndarray, blur_sigma: float = 2.2) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return binary
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    softened = cv2.GaussianBlur(binary, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.where(softened > 20, 255, 0).astype(np.uint8)


def _convex_region_mask(
    image_shape: tuple[int, int, int],
    point_groups: list[np.ndarray],
    blur_sigma: float = 2.2,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.concatenate(point_groups, axis=0).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return _soften_binary_mask(mask, blur_sigma=blur_sigma)


def _circle_mask(
    image_shape: tuple[int, int, int],
    center: np.ndarray,
    radius: int,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x, y = np.round(center).astype(int)
    cv2.circle(mask, (x, y), max(1, int(radius)), 255, thickness=-1)
    return _soften_binary_mask(mask, blur_sigma=blur_sigma)


def _line_corridor_mask(
    image_shape: tuple[int, int, int],
    start: np.ndarray,
    end: np.ndarray,
    thickness: int,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    start_xy = tuple(np.round(start).astype(int))
    end_xy = tuple(np.round(end).astype(int))
    cv2.line(mask, start_xy, end_xy, 255, thickness=max(1, int(thickness)))
    return _soften_binary_mask(mask, blur_sigma=blur_sigma)


def build_disease_edit_mask(
    image_shape: tuple[int, int, int],
    landmarks: np.ndarray,
    disease_mask: np.ndarray,
    disease: str,
    severity: float,
) -> np.ndarray:
    if cv2.countNonZero(disease_mask) == 0:
        return disease_mask.copy()

    severity_scale = float(np.clip(severity, 0.0, 1.5) / 1.5)
    disease = disease.strip().lower().replace(" ", "_")
    specialized = np.zeros(image_shape[:2], dtype=np.uint8)

    if disease in {"chin_deviation_left", "chin_deviation_right"}:
        is_left = disease.endswith("_left")
        dominant_jaw = LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX if is_left else RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX
        secondary_jaw = RIGHT_LOWER_JAW_IDX if is_left else LEFT_LOWER_JAW_IDX
        lower_lip_mid = landmarks[[17, 14]].mean(axis=0)
        chin_tip = landmarks[[152, 377]].mean(axis=0)
        mouth_mid = landmarks[MOUTH_CORNER_IDX].mean(axis=0)
        dominant_corner = landmarks[MOUTH_CORNER_IDX[0 if is_left else 1]]
        chin_core = CHIN_IDX + LOWER_LIP_IDX[:6]
        primary = _convex_region_mask(
            image_shape,
            [
                landmarks[dominant_jaw],
                landmarks[CHIN_IDX],
                landmarks[LOWER_LIP_IDX],
            ],
            blur_sigma=2.4,
        )
        secondary = _convex_region_mask(
            image_shape,
            [
                landmarks[secondary_jaw],
                landmarks[CHIN_IDX],
            ],
            blur_sigma=2.8,
        )
        core = _convex_region_mask(
            image_shape,
            [
                landmarks[chin_core],
                landmarks[MOUTH_CORNER_IDX],
            ],
            blur_sigma=2.0,
        )
        central_corridor = _line_corridor_mask(
            image_shape,
            start=chin_tip,
            end=lower_lip_mid,
            thickness=max(20, int(round(image_shape[0] * 0.022))),
            blur_sigma=2.0,
        )
        lip_mid_focus = _circle_mask(
            image_shape,
            center=lower_lip_mid,
            radius=max(16, int(round(image_shape[0] * 0.016))),
            blur_sigma=1.8,
        )
        chin_focus = _circle_mask(
            image_shape,
            center=chin_tip,
            radius=max(18, int(round(image_shape[0] * 0.018))),
            blur_sigma=1.8,
        )
        dominant_corner_focus = _circle_mask(
            image_shape,
            center=dominant_corner,
            radius=max(16, int(round(image_shape[0] * 0.016))),
            blur_sigma=2.0,
        )
        specialized = np.clip(
            primary.astype(np.float32) * (0.90 + 0.10 * severity_scale)
            + secondary.astype(np.float32) * (0.28 + 0.10 * severity_scale)
            + core.astype(np.float32) * 0.45
            + central_corridor.astype(np.float32) * 1.15
            + lip_mid_focus.astype(np.float32) * 1.00
            + chin_focus.astype(np.float32) * 1.00
            + dominant_corner_focus.astype(np.float32) * 0.35,
            0.0,
            255.0,
        ).astype(np.uint8)
    elif disease in {"occlusal_plane_cant", "occlusal_plane_cant_right"}:
        left_corner = landmarks[MOUTH_CORNER_IDX[0]]
        right_corner = landmarks[MOUTH_CORNER_IDX[1]]
        mouth_mid = (left_corner + right_corner) * 0.5
        lower_lip_mid = landmarks[[17, 14]].mean(axis=0)
        mouth_band = _convex_region_mask(
            image_shape,
            [
                landmarks[MOUTH_OUTER_IDX],
                landmarks[LOWER_LIP_IDX],
                landmarks[MOUTH_CORNER_IDX],
            ],
            blur_sigma=2.0,
        )
        lower_face = _convex_region_mask(
            image_shape,
            [
                landmarks[LEFT_LOWER_JAW_IDX],
                landmarks[RIGHT_LOWER_JAW_IDX],
                landmarks[CHIN_IDX],
            ],
            blur_sigma=3.0,
        )
        commissure_corridor = _line_corridor_mask(
            image_shape,
            start=left_corner,
            end=right_corner,
            thickness=max(18, int(round(image_shape[0] * 0.018))),
            blur_sigma=2.0,
        )
        vertical_compensation = _line_corridor_mask(
            image_shape,
            start=mouth_mid,
            end=lower_lip_mid,
            thickness=max(16, int(round(image_shape[0] * 0.016))),
            blur_sigma=2.0,
        )
        corner_focus = np.zeros(image_shape[:2], dtype=np.float32)
        for corner in (left_corner, right_corner):
            corner_focus += _circle_mask(
                image_shape,
                center=corner,
                radius=max(18, int(round(image_shape[0] * 0.018))),
                blur_sigma=2.2,
            ).astype(np.float32)
        corner_focus = np.clip(corner_focus, 0.0, 255.0).astype(np.uint8)
        specialized = np.clip(
            mouth_band.astype(np.float32) * (0.60 + 0.08 * severity_scale)
            + lower_face.astype(np.float32) * (0.10 + 0.05 * severity_scale)
            + commissure_corridor.astype(np.float32) * 1.15
            + vertical_compensation.astype(np.float32) * 0.55
            + corner_focus.astype(np.float32) * 1.30,
            0.0,
            255.0,
        ).astype(np.uint8)
    else:
        specialized = disease_mask.copy()

    combined = cv2.bitwise_and(_soften_binary_mask(specialized, blur_sigma=2.2), disease_mask)
    if cv2.countNonZero(combined) == 0:
        return disease_mask.copy()
    return combined


def resolve_sd_device_and_dtype(requested_device: str) -> tuple[str, torch.dtype]:
    requested = requested_device.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Stable Diffusion refinement, but no CUDA device is available.")
        return "cuda", torch.float16
    return "cpu", torch.float32


def pil_from_bgr(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def set_sd_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sd_inpaint_pipeline(model_id: str, device: str, dtype: torch.dtype) -> StableDiffusionInpaintPipeline:
    if StableDiffusionInpaintPipeline is None:
        raise RuntimeError("diffusers inpaint pipeline is not available, so local edit refinement cannot run.")
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Stable Diffusion inpaint model '{model_id}' from the local cache. "
            "Download the model first or point --sd-model-id to a local directory."
        ) from exc
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe.to(device)
    return pipe


def refine_face_with_sd_local_edit(
    original_image: np.ndarray,
    geometric_image: np.ndarray,
    landmarks: np.ndarray,
    disease_mask: np.ndarray,
    disease: str,
    severity: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if cv2.countNonZero(disease_mask) == 0:
        return geometric_image.copy(), {"enabled": False, "reason": "empty_disease_mask", "mode": "local_edit"}, None, None, None

    device, dtype = resolve_sd_device_and_dtype(args.sd_device)
    set_sd_seed(args.sd_seed)
    pipe = load_sd_inpaint_pipeline(args.sd_model_id, device, dtype)
    prompt = build_sd_prompt(disease, severity, args.sd_prompt)
    negative_prompt = args.sd_negative_prompt.strip()
    edit_mask = build_disease_edit_mask(
        image_shape=geometric_image.shape,
        landmarks=landmarks,
        disease_mask=disease_mask,
        disease=disease,
        severity=severity,
    )
    bbox = compute_edit_bbox(edit_mask, geometric_image.shape, pad_ratio=float(args.sd_roi_pad_ratio))
    x_min, y_min, x_max, y_max = bbox
    mask_crop = edit_mask[y_min:y_max, x_min:x_max]
    original_crop = original_image[y_min:y_max, x_min:x_max]
    geometric_crop = geometric_image[y_min:y_max, x_min:x_max]
    if args.sd_render_main:
        init_crop = build_renderer_condition_image(original_crop, geometric_crop, mask_crop)
    else:
        init_crop = geometric_crop.copy()

    init_image = pil_from_bgr(init_crop)
    mask_image = Image.fromarray(mask_crop).convert("L")
    generator = torch.Generator(device=device).manual_seed(int(args.sd_seed))

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            strength=float(np.clip(args.sd_strength, 0.0, 1.0)),
            guidance_scale=float(args.sd_guidance_scale),
            num_inference_steps=int(args.sd_steps),
            generator=generator,
        )

    refined_crop = bgr_from_pil(output.images[0])
    if refined_crop.shape[:2] != init_crop.shape[:2]:
        refined_crop = cv2.resize(refined_crop, (init_crop.shape[1], init_crop.shape[0]), interpolation=cv2.INTER_CUBIC)

    soft_mask = cv2.GaussianBlur((mask_crop.astype(np.float32) / 255.0), (0, 0), sigmaX=2.6, sigmaY=2.6)
    soft_mask_3 = np.repeat(np.clip(soft_mask, 0.0, 1.0)[:, :, None], 3, axis=2)
    blended_crop = refined_crop.astype(np.float32) * soft_mask_3 + geometric_crop.astype(np.float32) * (1.0 - soft_mask_3)
    blended_crop = np.clip(blended_crop, 0, 255).astype(np.uint8)

    refined_full = geometric_image.copy()
    refined_full[y_min:y_max, x_min:x_max] = blended_crop
    meta = {
        "enabled": True,
        "mode": "local_edit",
        "model_id": args.sd_model_id,
        "device": device,
        "steps": int(args.sd_steps),
        "guidance_scale": float(args.sd_guidance_scale),
        "strength": float(np.clip(args.sd_strength, 0.0, 1.0)),
        "seed": int(args.sd_seed),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "mask_pixels": int(cv2.countNonZero(edit_mask)),
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
    }
    condition_crop = init_crop if args.sd_render_main else None
    return refined_full, meta, condition_crop, refined_crop, edit_mask
