from __future__ import annotations
# このファイルの役割:
# 口 ROI ベースで歯科疾患の見た目を確認する主力スクリプトです。
# `protrusion`、`openbite`、`spacing`、`caries` の挙動確認に使います。

"""口ROIベースで疾患テンプレを当てる現在の主力スクリプト。

学習やDiffusionは使わず、
顔画像 -> MediaPipe口検出 -> 歯マスク推定 -> 局所warp -> 合成
の流れだけを確認するための入口。
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers import StableDiffusionInpaintPipeline
except Exception:
    StableDiffusionImg2ImgPipeline = None  # type: ignore[assignment]
    StableDiffusionInpaintPipeline = None  # type: ignore[assignment]


MOUTH_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
]
INNER_MOUTH_IDX = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

UPPER_INNER_IDX = [78, 95, 88, 87]
LOWER_INNER_IDX = [317, 402, 318, 324]
LEFT_FRONT_IDX = [78, 95, 88]
RIGHT_FRONT_IDX = [318, 324]
DEFAULT_FACE_LANDMARKER_PATH = Path("models") / "face_landmarker.task"
SUPPORTED_DISEASE_CHOICES = [
    "protrusion",
    "maxillary_protrusion",
    "openbite",
    "open_bite",
    "spacing",
    "diastema",
    "caries",
    "dental_caries",
    "dental caries",
]

DEFAULT_SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_SD_PROMPT = (
    "realistic teeth, natural dental appearance, clinical dental photo, "
    "realistic oral cavity, high detail, preserve the same person, preserve mouth structure"
)
DEFAULT_SD_NEGATIVE_PROMPT = (
    "extra teeth, blurry teeth, malformed mouth, deformed face, duplicate teeth, ugly, "
    "cartoon, illustration, painting, cgi, 3d render, stylized, altered identity, different person"
)

''' 指定された顔画像に対して、選択された疾患テンプレートに基づいて顔の形を歪ませる処理を実行するためのコマンドライン引数を定義する関数 '''
def normalize_template_disease_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    alias_map = {
        "protrusion": "protrusion",
        "maxillary_protrusion": "protrusion",
        "openbite": "openbite",
        "open_bite": "openbite",
        "anterior_open_bite": "openbite",
        "spacing": "spacing",
        "diastema": "spacing",
        "caries": "caries",
        "dental_caries": "caries",
    }
    if normalized not in alias_map:
        available = ", ".join(SUPPORTED_DISEASE_CHOICES)
        raise KeyError(f"Unsupported disease template: {name}. Available: {available}")
    return alias_map[normalized]

'''複数の歪み強度で処理した結果を横に並べた画像を生成する関数 '''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mouth-only geometric disease template inference.")
    parser.add_argument("--input", required=True, help="Input face image path")
    parser.add_argument(
        "--disease",
        required=True,
        choices=SUPPORTED_DISEASE_CHOICES,
        help="Template name",
    )
    parser.add_argument("--severity", type=float, default=0.5, help="Strength in [0, 1.5]")
    parser.add_argument(
        "--severity-sweep",
        default="",
        help="Comma-separated severities, e.g. 0.0,0.4,0.8,1.2. When set, runs all severities.",
    )
    parser.add_argument("--margin", type=int, default=20, help="Mouth ROI padding in pixels")
    parser.add_argument(
        "--face-landmarker-model",
        default=str(DEFAULT_FACE_LANDMARKER_PATH),
        help="Path to MediaPipe Face Landmarker .task model",
    )
    parser.add_argument("--output-dir", default="outputs/mouth_template", help="Directory for saved outputs")
    parser.add_argument("--enable-sd-refine", action="store_true", help="Run Stable Diffusion img2img on the warped mouth ROI")
    parser.add_argument("--sd-model-id", default=DEFAULT_SD_MODEL_ID, help="Stable Diffusion model id or local directory")
    parser.add_argument("--sd-device", default="auto", help="Diffusion device: auto/cuda/cpu")
    parser.add_argument("--sd-steps", type=int, default=24, help="Img2img inference steps")
    parser.add_argument("--sd-guidance-scale", type=float, default=4.5, help="Img2img guidance scale")
    parser.add_argument("--sd-strength", type=float, default=0.2, help="Img2img denoise strength")
    parser.add_argument("--sd-seed", type=int, default=1234, help="Img2img random seed")
    parser.add_argument("--sd-prompt", default="", help="Optional full positive prompt override")
    parser.add_argument("--sd-negative-prompt", default=DEFAULT_SD_NEGATIVE_PROMPT, help="Negative prompt for img2img")
    parser.add_argument("--sd-local-edit", action="store_true", help="Use disease-local inpaint refinement instead of full ROI img2img")
    parser.add_argument("--sd-render-main", action="store_true", help="Treat the geometric output as a condition image and let inpaint do the final rendering")
    return parser.parse_args()

'''単一の歪み強度で顔変形処理を実行し、結果を保存する関数'''
def heuristic_landmarks(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    face_x = int(w * 0.18)
    face_y = int(h * 0.08)
    face_w = int(w * 0.64)
    face_h = int(h * 0.84)
    mouth_center = np.array([face_x + face_w * 0.5, face_y + face_h * 0.78], dtype=np.float32)
    mouth_center[1] = max(mouth_center[1], h * 0.68)
    outer_radius = np.array([face_w * 0.18, face_h * 0.06], dtype=np.float32)
    inner_radius = outer_radius * np.array([0.72, 0.58], dtype=np.float32)

    landmarks = np.zeros((468, 2), dtype=np.int32)
    for offset, idx in enumerate(MOUTH_IDX):
        angle = (2.0 * np.pi * offset) / len(MOUTH_IDX)
        radius = inner_radius if idx in {78, 95, 88, 178, 87, 14, 317, 402, 318, 324} else outer_radius
        point = mouth_center + np.array([np.cos(angle) * radius[0], np.sin(angle) * radius[1]], dtype=np.float32)
        landmarks[idx] = np.round(point).astype(np.int32)
    return landmarks

'''あらかじめ用意した型に合わせて顔の形を歪ませる処理と、
お好みで画像生成AI（Stable Diffusion）を使ってキレイに仕上げるための実行プログラム'''
def detect_landmarks_with_tasks(image: np.ndarray, model_path: Path) -> np.ndarray:
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_bytes = model_path.read_bytes()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected by MediaPipe FaceLandmarker.")

    h, w = image.shape[:2]
    coords: list[list[int]] = []
    for lm in result.face_landmarks[0]:
        coords.append([int(round(lm.x * w)), int(round(lm.y * h))])
    return np.array(coords, dtype=np.int32)

'''コマンドライン引数を処理し、顔変形処理を実行して結果を保存するメイン関数'''
def load_image_unicode_safe(path: str) -> np.ndarray | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    binary = np.fromfile(str(image_path), dtype=np.uint8)
    if binary.size == 0:
        return None
    return cv2.imdecode(binary, cv2.IMREAD_COLOR)

'''コマンドライン引数を処理し、顔変形処理を実行して結果を保存するメイン関数'''
def detect_landmarks(image: np.ndarray, face_landmarker_model: str) -> tuple[np.ndarray, str]:
    model_path = Path(face_landmarker_model)
    if model_path.exists():
        try:
            return detect_landmarks_with_tasks(image, model_path), "mediapipe_tasks_face_landmarker"
        except Exception:
            pass

    face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_api is None:
        return heuristic_landmarks(image), "heuristic_fallback"

    with face_mesh_api.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return heuristic_landmarks(image), "heuristic_fallback"

    h, w = image.shape[:2]
    coords: list[list[int]] = []
    for lm in result.multi_face_landmarks[0].landmark:
        coords.append([int(round(lm.x * w)), int(round(lm.y * h))])
    return np.array(coords, dtype=np.int32), "mediapipe_facemesh"

'''複数の歪み強度で処理した結果を横に並べた画像を生成する関数'''
def extract_mouth_roi(image: np.ndarray, landmarks: np.ndarray, margin: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mouth_pts = landmarks[MOUTH_IDX]

    x_min = max(int(np.min(mouth_pts[:, 0])) - margin, 0)
    y_min = max(int(np.min(mouth_pts[:, 1])) - margin, 0)
    x_max = min(int(np.max(mouth_pts[:, 0])) + margin, image.shape[1])
    y_max = min(int(np.max(mouth_pts[:, 1])) + margin, image.shape[0])

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid mouth ROI computed from landmarks.")

    roi = image[y_min:y_max, x_min:x_max].copy()
    return roi, (x_min, y_min, x_max, y_max)


'''複数の歪み強度で処理した結果を横に並べた画像を生成する関数'''
def build_polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def build_teeth_mask(roi: np.ndarray, landmarks: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min, _, _ = bbox
    mouth_pts = landmarks[MOUTH_IDX] - np.array([x_min, y_min], dtype=np.int32)
    inner_pts = landmarks[INNER_MOUTH_IDX] - np.array([x_min, y_min], dtype=np.int32)

    outer_mask = build_polygon_mask(roi.shape[:2], mouth_pts)
    inner_mask = build_polygon_mask(roi.shape[:2], inner_pts)
    if cv2.countNonZero(inner_mask) == 0:
        inner_mask = outer_mask.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    value = hsv[:, :, 2].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    a_channel = lab[:, :, 1].astype(np.float32)

    inner_pixels = inner_mask > 0
    if not np.any(inner_pixels):
        inner_pixels = outer_mask > 0

    tooth_v = np.percentile(value[inner_pixels], 65)
    tooth_s = np.percentile(sat[inner_pixels], 55)
    gum_a = np.percentile(a_channel[inner_pixels], 55)
    teeth_binary = ((value >= tooth_v) & (sat <= tooth_s) & (a_channel <= gum_a + 4.0) & inner_pixels).astype(np.uint8) * 255
    teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_teeth_area = max(24, int(np.count_nonzero(inner_mask) * 0.08))
    if cv2.countNonZero(teeth_binary) < min_teeth_area:
        fallback = cv2.bitwise_and(inner_mask, cv2.inRange(hsv[:, :, 2], int(tooth_v), 255))
        teeth_binary = cv2.bitwise_or(teeth_binary, fallback)
        teeth_binary = cv2.morphologyEx(teeth_binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    teeth_binary = cv2.dilate(teeth_binary, np.ones((3, 3), np.uint8), iterations=1)
    teeth_binary = cv2.bitwise_and(teeth_binary, inner_mask)

    debug = roi.copy()
    contours, _ = cv2.findContours(teeth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (0, 255, 255), 1)
    cv2.polylines(debug, [mouth_pts.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)
    cv2.polylines(debug, [inner_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
    return teeth_binary, debug


def normalize_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma, sigmaY=sigma)
    max_value = float(soft.max())
    if max_value > 1e-6:
        soft = soft / max_value
    return np.clip(soft, 0.0, 1.0)


def build_teeth_geometry(teeth_mask: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    h, w = teeth_mask.shape
    debug = np.zeros((h, w, 3), dtype=np.uint8)
    masks: dict[str, np.ndarray] = {}
    if cv2.countNonZero(teeth_mask) == 0:
        zeros = np.zeros((h, w), dtype=np.float32)
        return {
            "full": zeros,
            "upper": zeros,
            "lower": zeros,
            "left": zeros,
            "right": zeros,
            "center": zeros,
            "upper_center": zeros,
            "left_central": zeros,
            "right_central": zeros,
            "gap_core": zeros,
            "front_pair": zeros,
        }, debug

    ys, xs = np.where(teeth_mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    split_x = float(np.median(xs))
    split_y = float(np.median(ys))
    center_x = (x_min + x_max) * 0.5
    center_y = (y_min + y_max) * 0.5
    band_sigma_x = max(4.0, (x_max - x_min) * 0.16)
    band_sigma_y = max(3.0, (y_max - y_min) * 0.20)
    central_sigma_x = max(3.0, (x_max - x_min) * 0.10)
    upper_sigma_y = max(3.0, (y_max - y_min) * 0.16)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    full_mask = normalize_mask(teeth_mask, sigma=3.0)
    upper_binary = np.where((grid_y <= split_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    lower_binary = np.where((grid_y > split_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    left_binary = np.where((grid_x < split_x) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    right_binary = np.where((grid_x >= split_x) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    center_band = np.exp(-((grid_x - center_x) ** 2) / (2.0 * band_sigma_x ** 2))
    tight_center_band = np.exp(-((grid_x - center_x) ** 2) / (2.0 * central_sigma_x ** 2))
    upper_focus = np.exp(-((grid_y - (center_y - band_sigma_y * 0.55)) ** 2) / (2.0 * upper_sigma_y ** 2))
    center_binary = np.where((center_band > 0.25) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    upper_center_binary = np.where((center_band > 0.18) & (grid_y <= center_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    left_central_binary = np.where(
        (tight_center_band > 0.30) & (upper_focus > 0.22) & (grid_x <= center_x) & (upper_binary > 0),
        255,
        0,
    ).astype(np.uint8)
    right_central_binary = np.where(
        (tight_center_band > 0.30) & (upper_focus > 0.22) & (grid_x >= center_x) & (upper_binary > 0),
        255,
        0,
    ).astype(np.uint8)
    gap_core_binary = np.where(
        (tight_center_band > 0.55) & (upper_focus > 0.18) & (teeth_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    masks["full"] = full_mask
    masks["upper"] = normalize_mask(upper_binary, sigma=3.0)
    masks["lower"] = normalize_mask(lower_binary, sigma=3.0)
    masks["left"] = normalize_mask(left_binary, sigma=3.0)
    masks["right"] = normalize_mask(right_binary, sigma=3.0)
    masks["center"] = normalize_mask(center_binary, sigma=3.0)
    masks["upper_center"] = normalize_mask(upper_center_binary, sigma=3.0)
    masks["left_central"] = normalize_mask(left_central_binary, sigma=2.0)
    masks["right_central"] = normalize_mask(right_central_binary, sigma=2.0)
    masks["gap_core"] = normalize_mask(gap_core_binary, sigma=1.8)
    masks["front_pair"] = np.clip(masks["left_central"] + masks["right_central"], 0.0, 1.0)

    debug[:, :, 1] = (masks["full"] * 255.0).astype(np.uint8)
    debug[:, :, 0] = np.clip((masks["left_central"] + masks["right_central"]) * 255.0, 0, 255).astype(np.uint8)
    debug[:, :, 2] = np.clip(masks["gap_core"] * 255.0, 0, 255).astype(np.uint8)
    cv2.line(debug, (int(round(split_x)), y_min), (int(round(split_x)), y_max), (255, 255, 255), 1)
    cv2.line(debug, (x_min, int(round(split_y))), (x_max, int(round(split_y))), (255, 255, 255), 1)
    return masks, debug


def smooth_displacement_field(delta_x: np.ndarray, delta_y: np.ndarray, sigma: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
    smoothed_x = cv2.GaussianBlur(delta_x.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    smoothed_y = cv2.GaussianBlur(delta_y.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return smoothed_x, smoothed_y


def build_roi_distance_falloff(teeth_mask: np.ndarray, roi_shape: tuple[int, ...]) -> np.ndarray:
    h, w = roi_shape[:2]
    if cv2.countNonZero(teeth_mask) == 0:
        return np.ones((h, w), dtype=np.float32)
    dilated = cv2.dilate(teeth_mask, np.ones((7, 7), np.uint8), iterations=1)
    distance = cv2.distanceTransform(cv2.bitwise_not(dilated), cv2.DIST_L2, 5)
    radius = max(6.0, min(h, w) * 0.18)
    falloff = np.exp(-(distance ** 2) / (2.0 * radius ** 2)).astype(np.float32)
    falloff[dilated > 0] = 1.0
    return np.clip(falloff, 0.0, 1.0)


def build_template_delta(
    roi_shape: tuple[int, ...],
    disease: str,
    severity: float,
    teeth_mask: np.ndarray | None = None,
    teeth_geometry: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    disease = normalize_template_disease_name(disease)
    h, w = roi_shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    delta_x = np.zeros_like(grid_x, dtype=np.float32)
    delta_y = np.zeros_like(grid_y, dtype=np.float32)

    strength = max(0.0, severity)
    base_shift = max(2.0, min(h, w) * 0.08 * strength)
    if teeth_mask is not None and cv2.countNonZero(teeth_mask) > 0:
        focus_mask = cv2.GaussianBlur(teeth_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=4.0, sigmaY=4.0)
    else:
        focus_mask = np.ones((h, w), dtype=np.float32)
    geometry = teeth_geometry or {}
    upper_mask = geometry.get("upper", focus_mask)
    lower_mask = geometry.get("lower", focus_mask)
    left_mask = geometry.get("left", focus_mask)
    right_mask = geometry.get("right", focus_mask)
    center_mask = geometry.get("center", focus_mask)
    upper_center_mask = geometry.get("upper_center", upper_mask)
    left_central_mask = geometry.get("left_central", upper_center_mask * left_mask)
    right_central_mask = geometry.get("right_central", upper_center_mask * right_mask)
    gap_core_mask = geometry.get("gap_core", center_mask * upper_center_mask)
    front_pair_mask = geometry.get("front_pair", np.clip(left_central_mask + right_central_mask, 0.0, 1.0))

    if disease == "protrusion":
        center_x = w * 0.5
        center_y = h * 0.48
        sigma_x = max(5.0, w * 0.10)
        sigma_y = max(4.0, h * 0.10)
        forward_focus = np.exp(-(((grid_x - center_x) ** 2) / (2.0 * sigma_x ** 2) + ((grid_y - center_y) ** 2) / (2.0 * sigma_y ** 2)))
        arch_compact = np.exp(-((grid_y - center_y) ** 2) / (2.0 * max(5.0, h * 0.14) ** 2))
        delta_y -= base_shift * 1.55 * forward_focus * upper_center_mask
        delta_x += base_shift * 0.42 * forward_focus * upper_center_mask
        delta_y -= base_shift * 0.45 * arch_compact * upper_mask
        delta_x -= base_shift * 0.12 * left_mask * upper_center_mask
        delta_x += base_shift * 0.12 * right_mask * upper_center_mask
    elif disease == "openbite":
        center_y = h * 0.52
        vertical_band = np.exp(-((grid_y - center_y) ** 2) / (2.0 * max(4.0, h * 0.12) ** 2))
        horizontal_focus = np.exp(-((grid_x - w * 0.5) ** 2) / (2.0 * max(6.0, w * 0.15) ** 2))
        gap_focus = vertical_band * horizontal_focus * center_mask
        delta_y -= base_shift * 1.65 * gap_focus * upper_mask
        delta_y += base_shift * 1.65 * gap_focus * lower_mask
    elif disease == "spacing":
        crown_focus = np.exp(-((grid_y - h * 0.46) ** 2) / (2.0 * max(4.0, h * 0.11) ** 2))
        root_focus = np.exp(-((grid_y - h * 0.58) ** 2) / (2.0 * max(4.0, h * 0.14) ** 2))
        tooth_rigidity = 0.35 + 0.65 * crown_focus
        gap_focus = gap_core_mask * crown_focus

        # Treat the two front incisors like separate pieces: left tooth goes left, right tooth goes right.
        delta_x -= base_shift * 4.25 * left_central_mask * tooth_rigidity
        delta_x += base_shift * 4.25 * right_central_mask * tooth_rigidity

        # Anchor the roots a little less than the crowns so the split reads like two teeth separating.
        delta_x += base_shift * 0.42 * left_central_mask * root_focus
        delta_x -= base_shift * 0.42 * right_central_mask * root_focus

        # Slight crown lift keeps the contact from visually smearing into a broad arch stretch.
        delta_y -= base_shift * 0.34 * front_pair_mask * crown_focus
        delta_y += base_shift * 0.08 * front_pair_mask * root_focus

        # Carve the midline more aggressively so SD has a clear gap to preserve.
        delta_x -= base_shift * 1.35 * gap_focus * left_central_mask
        delta_x += base_shift * 1.35 * gap_focus * right_central_mask
    elif disease == "caries":
        pass
    else:
        raise KeyError(f"Unsupported disease template: {disease}")

    if teeth_mask is not None:
        falloff = build_roi_distance_falloff(teeth_mask, roi_shape)
        delta_x *= falloff
        delta_y *= falloff
    return smooth_displacement_field(delta_x, delta_y)


def warp_mouth(roi: np.ndarray, delta_x: np.ndarray, delta_y: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + delta_x).astype(np.float32)
    map_y = (grid_y + delta_y).astype(np.float32)
    return cv2.remap(
        roi,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def shift_mask_and_patch(
    image: np.ndarray,
    mask_soft: np.ndarray,
    shift_x: int,
    shift_y: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask_soft.shape
    if shift_x == 0 and shift_y == 0:
        return image.copy(), mask_soft.copy()

    translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_patch = cv2.warpAffine(
        image,
        translation,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    shifted_mask = cv2.warpAffine(
        mask_soft.astype(np.float32),
        translation,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted_patch, np.clip(shifted_mask, 0.0, 1.0)


def apply_spacing_front_teeth_translation(
    roi: np.ndarray,
    severity: float,
    teeth_geometry: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = teeth_geometry or {}
    left_central = np.clip(geometry.get("left_central", np.zeros(roi.shape[:2], dtype=np.float32)), 0.0, 1.0)
    right_central = np.clip(geometry.get("right_central", np.zeros(roi.shape[:2], dtype=np.float32)), 0.0, 1.0)
    gap_core = np.clip(geometry.get("gap_core", np.zeros(roi.shape[:2], dtype=np.float32)), 0.0, 1.0)
    front_pair = np.clip(geometry.get("front_pair", left_central + right_central), 0.0, 1.0)
    upper_center = np.clip(geometry.get("upper_center", front_pair), 0.0, 1.0)

    if float(left_central.max()) < 1e-6 or float(right_central.max()) < 1e-6:
        return roi.copy(), np.zeros(roi.shape[:2], dtype=np.uint8)

    h, w = roi.shape[:2]
    severity_scale = float(np.clip(severity, 0.0, 1.5) / 1.5)
    shift_px = max(4, int(round((w * 0.018) + (w * 0.030 * severity_scale))))
    lift_px = max(0, int(round(h * 0.012 * severity_scale)))

    left_patch, left_shifted_mask = shift_mask_and_patch(roi, left_central, -shift_px, -lift_px)
    right_patch, right_shifted_mask = shift_mask_and_patch(roi, right_central, shift_px, -lift_px)

    # Remove the original two incisors from the base so they can be re-placed as separate pieces.
    front_removal = np.clip(front_pair * (0.68 + 0.32 * upper_center), 0.0, 1.0)
    front_removal = cv2.GaussianBlur(front_removal.astype(np.float32), (0, 0), sigmaX=2.2, sigmaY=2.2)
    gap_seed = cv2.GaussianBlur(np.clip(gap_core * (0.65 + 0.35 * upper_center), 0.0, 1.0), (0, 0), sigmaX=2.0, sigmaY=2.0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mouth_shadow = (
        0.50 * hsv[:, :, 2].astype(np.float32)
        + 0.30 * hsv[:, :, 1].astype(np.float32)
        + 0.20 * lab[:, :, 1].astype(np.float32)
    )
    mouth_shadow = cv2.GaussianBlur(mouth_shadow, (0, 0), sigmaX=4.0, sigmaY=4.0)
    mouth_shadow = cv2.normalize(mouth_shadow, None, 0.0, 1.0, cv2.NORM_MINMAX)

    dark_fill = roi.astype(np.float32).copy()
    dark_fill[:, :, 0] *= 0.48 + 0.10 * (1.0 - mouth_shadow)
    dark_fill[:, :, 1] *= 0.40 + 0.08 * (1.0 - mouth_shadow)
    dark_fill[:, :, 2] *= 0.34 + 0.06 * (1.0 - mouth_shadow)

    base = roi.astype(np.float32) * (1.0 - front_removal[:, :, None]) + dark_fill * front_removal[:, :, None]

    # Put the separated incisors back on top.
    base = base * (1.0 - left_shifted_mask[:, :, None]) + left_patch.astype(np.float32) * left_shifted_mask[:, :, None]
    base = base * (1.0 - right_shifted_mask[:, :, None]) + right_patch.astype(np.float32) * right_shifted_mask[:, :, None]

    # Explicit midline gap cue that survives the later diffusion refinement.
    gap_mask = cv2.GaussianBlur(
        np.clip(gap_seed + front_removal * 0.22, 0.0, 1.0).astype(np.float32),
        (0, 0),
        sigmaX=1.8,
        sigmaY=1.8,
    )
    gap_mask = np.clip(gap_mask * (0.55 + 0.45 * severity_scale), 0.0, 1.0)
    base[:, :, 0] *= 1.0 - 0.34 * gap_mask
    base[:, :, 1] *= 1.0 - 0.42 * gap_mask
    base[:, :, 2] *= 1.0 - 0.52 * gap_mask

    return np.clip(base, 0, 255).astype(np.uint8), np.clip(gap_mask * 255.0, 0, 255).astype(np.uint8)


def apply_disease_texture(
    roi: np.ndarray,
    disease: str,
    severity: float,
    teeth_mask: np.ndarray | None = None,
    teeth_geometry: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    disease = normalize_template_disease_name(disease)
    if disease == "spacing":
        if teeth_mask is None or cv2.countNonZero(teeth_mask) == 0:
            return roi.copy(), np.zeros(roi.shape[:2], dtype=np.uint8)
        return apply_spacing_front_teeth_translation(roi, severity, teeth_geometry=teeth_geometry)

    if disease != "caries":
        return roi.copy(), np.zeros(roi.shape[:2], dtype=np.uint8)

    if teeth_mask is None or cv2.countNonZero(teeth_mask) == 0:
        return roi.copy(), np.zeros(roi.shape[:2], dtype=np.uint8)

    h, w = roi.shape[:2]
    geometry = teeth_geometry or {}
    center_mask = geometry.get("center", teeth_mask.astype(np.float32) / 255.0)
    upper_mask = geometry.get("upper", teeth_mask.astype(np.float32) / 255.0)
    base_mask = cv2.GaussianBlur(teeth_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=3.2, sigmaY=3.2)
    severity_scale = float(np.clip(severity, 0.0, 1.5) / 1.5)

    rng = np.random.default_rng(7)
    coarse_noise = rng.random((h, w), dtype=np.float32)
    coarse_noise = cv2.GaussianBlur(coarse_noise, (0, 0), sigmaX=max(3.0, w * 0.08), sigmaY=max(3.0, h * 0.08))
    coarse_noise = cv2.normalize(coarse_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

    edge_noise = rng.random((h, w), dtype=np.float32)
    edge_noise = cv2.GaussianBlur(edge_noise, (0, 0), sigmaX=max(1.8, w * 0.03), sigmaY=max(1.8, h * 0.03))
    edge_noise = cv2.normalize(edge_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

    center_bias = cv2.GaussianBlur((0.65 * center_mask + 0.35 * upper_mask).astype(np.float32), (0, 0), sigmaX=4.0, sigmaY=4.0)
    stain_strength = np.clip((0.35 + 0.65 * coarse_noise) * (0.55 + 0.45 * center_bias) * base_mask * severity_scale, 0.0, 1.0)
    cavity_mask = ((edge_noise > (0.74 - 0.18 * severity_scale)) & (base_mask > 0.18)).astype(np.float32)
    cavity_mask = cv2.GaussianBlur(cavity_mask, (0, 0), sigmaX=1.6, sigmaY=1.6)
    cavity_mask = np.clip(cavity_mask * base_mask, 0.0, 1.0)

    output = roi.astype(np.float32).copy()
    output[:, :, 0] *= 1.0 - 0.42 * stain_strength - 0.22 * cavity_mask
    output[:, :, 1] *= 1.0 - 0.28 * stain_strength - 0.16 * cavity_mask
    output[:, :, 2] = output[:, :, 2] * (1.0 - 0.06 * stain_strength) + 26.0 * stain_strength
    output *= 1.0 - 0.18 * cavity_mask[:, :, None]

    texture_mask = np.clip(np.maximum(stain_strength, cavity_mask) * 255.0, 0, 255).astype(np.uint8)
    return np.clip(output, 0, 255).astype(np.uint8), texture_mask


def build_blend_mask(roi_shape: tuple[int, ...], teeth_mask: np.ndarray | None = None) -> np.ndarray:
    h, w = roi_shape[:2]
    if teeth_mask is not None and cv2.countNonZero(teeth_mask) > 0:
        dilated = cv2.dilate(teeth_mask, np.ones((9, 9), np.uint8), iterations=1)
        mask = build_roi_distance_falloff(dilated, roi_shape)
        blur = max(9, int(round(min(h, w) * 0.16)))
    else:
        mask = np.zeros((h, w), dtype=np.float32)
        pad_x = max(6, int(round(w * 0.12)))
        pad_y = max(6, int(round(h * 0.18)))
        cv2.rectangle(mask, (pad_x, pad_y), (max(pad_x + 1, w - pad_x), max(pad_y + 1, h - pad_y)), 1.0, thickness=-1)
        blur = max(5, int(round(min(h, w) * 0.18)))
    if blur % 2 == 0:
        blur += 1
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def build_disease_edit_mask(
    roi_shape: tuple[int, ...],
    disease: str,
    severity: float,
    teeth_geometry: dict[str, np.ndarray] | None = None,
    texture_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = roi_shape[:2]
    geometry = teeth_geometry or {}
    zeros = np.zeros((h, w), dtype=np.float32)
    severity_scale = float(np.clip(severity, 0.0, 1.5) / 1.5)

    if disease == "spacing":
        gap_core = geometry.get("gap_core", zeros)
        left_central = geometry.get("left_central", zeros)
        right_central = geometry.get("right_central", zeros)
        upper_center = geometry.get("upper_center", zeros)
        mask = np.clip(gap_core * 1.35 + (left_central + right_central) * 0.32 + upper_center * 0.18, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=2.8, sigmaY=2.8)
    elif disease == "openbite":
        center = geometry.get("center", zeros)
        upper = geometry.get("upper", zeros)
        lower = geometry.get("lower", zeros)
        mask = np.clip(center * 0.85 + upper * 0.24 + lower * 0.24, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3.5, sigmaY=3.5)
    elif disease == "protrusion":
        upper_center = geometry.get("upper_center", zeros)
        center = geometry.get("center", zeros)
        mask = np.clip(upper_center * 0.95 + center * 0.20, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=3.2, sigmaY=3.2)
    elif disease == "caries":
        if texture_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        mask = cv2.GaussianBlur((texture_mask.astype(np.float32) / 255.0), (0, 0), sigmaX=2.4, sigmaY=2.4)
    else:
        mask = zeros

    threshold = 0.10 + (0.06 * (1.0 - min(1.0, severity_scale)))
    binary = (mask > threshold).astype(np.uint8) * 255
    if cv2.countNonZero(binary) > 0:
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        binary = cv2.GaussianBlur(binary, (0, 0), sigmaX=1.6, sigmaY=1.6)
        binary = np.where(binary > 16, 255, 0).astype(np.uint8)
    return binary


def build_renderer_condition_roi(
    original_roi: np.ndarray,
    geometric_roi: np.ndarray,
    disease_mask: np.ndarray,
) -> np.ndarray:
    if cv2.countNonZero(disease_mask) == 0:
        return geometric_roi.copy()
    local_mask = cv2.GaussianBlur((disease_mask.astype(np.float32) / 255.0), (0, 0), sigmaX=2.6, sigmaY=2.6)
    local_mask_3 = np.repeat(np.clip(local_mask, 0.0, 1.0)[:, :, None], 3, axis=2)
    blended = original_roi.astype(np.float32) * (1.0 - local_mask_3) + geometric_roi.astype(np.float32) * local_mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


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
        "spacing": "diastema between upper incisors, realistic tooth spacing",
        "protrusion": "maxillary protrusion, orthodontic deformation, forward upper incisors",
        "openbite": "anterior open bite, visible gap between upper and lower front teeth",
        "caries": "dental caries, tooth decay, dark enamel lesion, realistic decayed tooth texture",
    }
    bucket = severity_to_prompt_bucket(severity)
    bucket_text = {
        "mild": "subtle medically plausible change",
        "moderate": "clear medically plausible change",
        "severe": "pronounced but realistic medically plausible change",
    }[bucket]
    disease_text = disease_prompts.get(disease, "realistic dental condition")
    return f"{DEFAULT_SD_PROMPT}, {disease_text}, {bucket_text}"


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


def load_sd_img2img_pipeline(model_id: str, device: str, dtype: torch.dtype) -> StableDiffusionImg2ImgPipeline:
    if StableDiffusionImg2ImgPipeline is None:
        raise RuntimeError("diffusers is not available, so Stable Diffusion img2img refinement cannot run.")
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Stable Diffusion model '{model_id}' from the local cache. "
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


def refine_roi_with_sd(
    warped_roi: np.ndarray,
    disease: str,
    severity: float,
    teeth_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    device, dtype = resolve_sd_device_and_dtype(args.sd_device)
    set_sd_seed(args.sd_seed)
    pipe = load_sd_img2img_pipeline(args.sd_model_id, device, dtype)
    prompt = build_sd_prompt(disease, severity, args.sd_prompt)
    negative_prompt = args.sd_negative_prompt.strip()
    init_image = pil_from_bgr(warped_roi)
    generator = torch.Generator(device=device).manual_seed(int(args.sd_seed))

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=float(np.clip(args.sd_strength, 0.0, 1.0)),
            guidance_scale=float(args.sd_guidance_scale),
            num_inference_steps=int(args.sd_steps),
            generator=generator,
        )
    refined_roi = bgr_from_pil(output.images[0])
    if refined_roi.shape[:2] != warped_roi.shape[:2]:
        refined_roi = cv2.resize(refined_roi, (warped_roi.shape[1], warped_roi.shape[0]), interpolation=cv2.INTER_CUBIC)

    preserve_mask = build_blend_mask(warped_roi.shape, teeth_mask=teeth_mask)
    preserve_mask_3 = np.repeat(preserve_mask[:, :, None], 3, axis=2)
    blended_refined = refined_roi.astype(np.float32) * preserve_mask_3 + warped_roi.astype(np.float32) * (1.0 - preserve_mask_3)
    blended_refined = np.clip(blended_refined, 0, 255).astype(np.uint8)
    meta = {
        "enabled": True,
        "model_id": args.sd_model_id,
        "device": device,
        "steps": int(args.sd_steps),
        "guidance_scale": float(args.sd_guidance_scale),
        "strength": float(np.clip(args.sd_strength, 0.0, 1.0)),
        "seed": int(args.sd_seed),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }
    return blended_refined, meta


def refine_roi_with_sd_local_edit(
    init_roi: np.ndarray,
    disease_mask: np.ndarray,
    disease: str,
    severity: float,
    teeth_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    if cv2.countNonZero(disease_mask) == 0:
        return init_roi.copy(), {"enabled": False, "reason": "empty_disease_mask", "mode": "local_edit"}

    device, dtype = resolve_sd_device_and_dtype(args.sd_device)
    set_sd_seed(args.sd_seed)
    pipe = load_sd_inpaint_pipeline(args.sd_model_id, device, dtype)
    prompt = build_sd_prompt(disease, severity, args.sd_prompt)
    negative_prompt = args.sd_negative_prompt.strip()
    init_image = pil_from_bgr(init_roi)
    mask_image = Image.fromarray(disease_mask).convert("L")
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
    refined_roi = bgr_from_pil(output.images[0])
    if refined_roi.shape[:2] != init_roi.shape[:2]:
        refined_roi = cv2.resize(refined_roi, (init_roi.shape[1], init_roi.shape[0]), interpolation=cv2.INTER_CUBIC)

    local_mask = cv2.GaussianBlur((disease_mask.astype(np.float32) / 255.0), (0, 0), sigmaX=2.2, sigmaY=2.2)
    if teeth_mask is not None and cv2.countNonZero(teeth_mask) > 0:
        local_mask = np.clip(local_mask * (0.55 + 0.45 * build_blend_mask(init_roi.shape, teeth_mask=teeth_mask)), 0.0, 1.0)
    local_mask_3 = np.repeat(local_mask[:, :, None], 3, axis=2)
    blended_refined = refined_roi.astype(np.float32) * local_mask_3 + init_roi.astype(np.float32) * (1.0 - local_mask_3)
    blended_refined = np.clip(blended_refined, 0, 255).astype(np.uint8)
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
        "mask_pixels": int(cv2.countNonZero(disease_mask)),
    }
    return blended_refined, meta


def blend_back(
    image: np.ndarray,
    warped_roi: np.ndarray,
    bbox: tuple[int, int, int, int],
    teeth_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, y_min, x_max, y_max = bbox
    result = image.copy()
    mask = build_blend_mask(warped_roi.shape, teeth_mask=teeth_mask)
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

    original = result[y_min:y_max, x_min:x_max].astype(np.float32)
    warped = warped_roi.astype(np.float32)
    blended = warped * mask_3 + original * (1.0 - mask_3)
    result[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)
    return result, (mask * 255.0).astype(np.uint8)


def draw_landmarks_debug(image: np.ndarray, landmarks: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    canvas = image.copy()
    mouth_pts = landmarks[MOUTH_IDX]
    for x, y in mouth_pts:
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
    if len(mouth_pts) > 2:
        hull = cv2.convexHull(mouth_pts.astype(np.int32))
        cv2.polylines(canvas, [hull], isClosed=True, color=(0, 200, 255), thickness=1)
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return canvas


def save_outputs(
    output_dir: Path,
    mouth_roi: np.ndarray,
    warped_roi: np.ndarray,
    condition_roi: np.ndarray | None,
    refined_roi: np.ndarray | None,
    final_output: np.ndarray,
    landmarks_debug: np.ndarray,
    blend_mask: np.ndarray,
    teeth_mask: np.ndarray,
    teeth_debug: np.ndarray,
    geometry_debug: np.ndarray,
    texture_mask: np.ndarray,
    disease_edit_mask: np.ndarray | None,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "mouth_roi.png"), mouth_roi)
    cv2.imwrite(str(output_dir / "warped_roi.png"), warped_roi)
    if condition_roi is not None:
        cv2.imwrite(str(output_dir / "condition_roi.png"), condition_roi)
    if refined_roi is not None:
        cv2.imwrite(str(output_dir / "refined_roi.png"), refined_roi)
    cv2.imwrite(str(output_dir / "final_output.png"), final_output)
    cv2.imwrite(str(output_dir / "landmarks_debug.png"), landmarks_debug)
    cv2.imwrite(str(output_dir / "blend_mask.png"), blend_mask)
    cv2.imwrite(str(output_dir / "teeth_mask.png"), teeth_mask)
    cv2.imwrite(str(output_dir / "teeth_debug.png"), teeth_debug)
    cv2.imwrite(str(output_dir / "geometry_debug.png"), geometry_debug)
    cv2.imwrite(str(output_dir / "texture_mask.png"), texture_mask)
    if disease_edit_mask is not None:
        cv2.imwrite(str(output_dir / "disease_edit_mask.png"), disease_edit_mask)

    delta_vis = np.zeros((delta_x.shape[0], delta_x.shape[1], 3), dtype=np.uint8)
    mag = np.sqrt(delta_x ** 2 + delta_y ** 2)
    if float(mag.max()) > 1e-6:
        mag_norm = np.clip(mag / mag.max(), 0.0, 1.0)
    else:
        mag_norm = mag
    delta_vis[:, :, 2] = (mag_norm * 255.0).astype(np.uint8)
    cv2.imwrite(str(output_dir / "delta_heatmap.png"), delta_vis)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_severity_values(args: argparse.Namespace) -> list[float]:
    if args.severity_sweep.strip():
        return [float(item.strip()) for item in args.severity_sweep.split(",") if item.strip()]
    return [float(args.severity)]


def render_severity_strip(output_paths: list[Path], save_path: Path) -> None:
    images: list[np.ndarray] = []
    for path in output_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        label = path.parent.name.replace("severity_", "s=")
        canvas = image.copy()
        cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(canvas)
    if images:
        strip = cv2.hconcat(images)
        cv2.imwrite(str(save_path), strip)


def run_single_case(image: np.ndarray, args: argparse.Namespace, severity_value: float, output_dir: Path) -> Path:
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    disease_name = normalize_template_disease_name(args.disease)
    landmarks, detector_mode = detect_landmarks(image, args.face_landmarker_model)
    mouth_roi, bbox = extract_mouth_roi(image, landmarks, margin=args.margin)
    teeth_mask, teeth_debug = build_teeth_mask(mouth_roi, landmarks, bbox)
    teeth_geometry, geometry_debug = build_teeth_geometry(teeth_mask)
    delta_x, delta_y = build_template_delta(
        mouth_roi.shape,
        disease_name,
        severity_value,
        teeth_mask=teeth_mask,
        teeth_geometry=teeth_geometry,
    )
    warped_roi = warp_mouth(mouth_roi, delta_x, delta_y)
    warped_roi, texture_mask = apply_disease_texture(
        warped_roi,
        disease_name,
        severity_value,
        teeth_mask=teeth_mask,
        teeth_geometry=teeth_geometry,
    )
    disease_edit_mask = build_disease_edit_mask(
        warped_roi.shape,
        disease_name,
        severity_value,
        teeth_geometry=teeth_geometry,
        texture_mask=texture_mask,
    )
    condition_roi: np.ndarray | None = None
    refined_roi: np.ndarray | None = None
    sd_meta: dict[str, object] = {"enabled": False}
    roi_for_blend = warped_roi
    if args.enable_sd_refine:
        if args.sd_local_edit:
            init_roi = warped_roi
            if args.sd_render_main:
                condition_roi = build_renderer_condition_roi(mouth_roi, warped_roi, disease_edit_mask)
                init_roi = condition_roi
            refined_roi, sd_meta = refine_roi_with_sd_local_edit(
                init_roi=init_roi,
                disease_mask=disease_edit_mask,
                disease=disease_name,
                severity=severity_value,
                teeth_mask=teeth_mask,
                args=args,
            )
        else:
            refined_roi, sd_meta = refine_roi_with_sd(
                warped_roi=warped_roi,
                disease=disease_name,
                severity=severity_value,
                teeth_mask=teeth_mask,
                args=args,
            )
        roi_for_blend = refined_roi

    final_output, blend_mask = blend_back(image, roi_for_blend, bbox, teeth_mask=teeth_mask)
    landmarks_debug = draw_landmarks_debug(image, landmarks, bbox)

    summary = {
        "input": str(Path(args.input).resolve()),
        "detector_mode": detector_mode,
        "disease": disease_name,
        "severity": severity_value,
        "bbox": {
            "x_min": bbox[0],
            "y_min": bbox[1],
            "x_max": bbox[2],
            "y_max": bbox[3],
        },
        "mouth_roi_shape": [int(v) for v in mouth_roi.shape],
        "teeth_pixels": int(cv2.countNonZero(teeth_mask)),
        "geometry_masks": sorted(list(teeth_geometry.keys())),
        "delta_abs_mean": float(np.mean(np.abs(delta_x)) + np.mean(np.abs(delta_y))),
        "delta_abs_max": float(max(np.max(np.abs(delta_x)), np.max(np.abs(delta_y)))),
        "texture_pixels": int(cv2.countNonZero(texture_mask)),
        "disease_edit_pixels": int(cv2.countNonZero(disease_edit_mask)),
        "sd_refine": sd_meta,
        "renderer_condition_enabled": bool(args.enable_sd_refine and args.sd_local_edit and args.sd_render_main),
    }

    save_outputs(
        output_dir=output_dir,
        mouth_roi=mouth_roi,
        warped_roi=warped_roi,
        condition_roi=condition_roi,
        refined_roi=refined_roi,
        final_output=final_output,
        landmarks_debug=landmarks_debug,
        blend_mask=blend_mask,
        teeth_mask=teeth_mask,
        teeth_debug=teeth_debug,
        geometry_debug=geometry_debug,
        texture_mask=texture_mask,
        disease_edit_mask=disease_edit_mask,
        delta_x=delta_x,
        delta_y=delta_y,
        summary=summary,
    )
    return output_dir / "final_output.png"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    image = load_image_unicode_safe(args.input)
    severity_values = parse_severity_values(args)
    rendered_paths: list[Path] = []
    for severity_value in severity_values:
        if len(severity_values) == 1:
            case_output_dir = output_root
        else:
            case_output_dir = output_root / f"severity_{severity_value:.2f}".replace(".", "_")
        rendered_paths.append(run_single_case(image, args, severity_value, case_output_dir))

    if len(rendered_paths) > 1:
        render_severity_strip(rendered_paths, output_root / "severity_strip.png")

    print(f"Done. Output saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
