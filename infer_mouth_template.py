from __future__ import annotations

"""口ROIベースで疾患テンプレを当てる現在の主力スクリプト。

学習やDiffusionは使わず、
顔画像 -> MediaPipe口検出 -> 歯マスク推定 -> 局所warp -> 合成
の流れだけを確認するための入口。
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import mediapipe as mp
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mouth-only geometric disease template inference.")
    parser.add_argument("--input", required=True, help="Input face image path")
    parser.add_argument(
        "--disease",
        required=True,
        choices=["protrusion", "openbite", "spacing"],
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
    return parser.parse_args()


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


def load_image_unicode_safe(path: str) -> np.ndarray | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    binary = np.fromfile(str(image_path), dtype=np.uint8)
    if binary.size == 0:
        return None
    return cv2.imdecode(binary, cv2.IMREAD_COLOR)


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

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    full_mask = normalize_mask(teeth_mask, sigma=3.0)
    upper_binary = np.where((grid_y <= split_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    lower_binary = np.where((grid_y > split_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    left_binary = np.where((grid_x < split_x) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    right_binary = np.where((grid_x >= split_x) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    center_band = np.exp(-((grid_x - center_x) ** 2) / (2.0 * band_sigma_x ** 2))
    center_binary = np.where((center_band > 0.25) & (teeth_mask > 0), 255, 0).astype(np.uint8)
    upper_center_binary = np.where((center_band > 0.18) & (grid_y <= center_y) & (teeth_mask > 0), 255, 0).astype(np.uint8)

    masks["full"] = full_mask
    masks["upper"] = normalize_mask(upper_binary, sigma=3.0)
    masks["lower"] = normalize_mask(lower_binary, sigma=3.0)
    masks["left"] = normalize_mask(left_binary, sigma=3.0)
    masks["right"] = normalize_mask(right_binary, sigma=3.0)
    masks["center"] = normalize_mask(center_binary, sigma=3.0)
    masks["upper_center"] = normalize_mask(upper_center_binary, sigma=3.0)

    debug[:, :, 1] = (masks["full"] * 255.0).astype(np.uint8)
    debug[:, :, 0] = (masks["upper"] * 255.0).astype(np.uint8)
    debug[:, :, 2] = (masks["center"] * 255.0).astype(np.uint8)
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
        center_x = w * 0.5
        split_focus = np.exp(-((grid_x - center_x) ** 2) / (2.0 * max(3.0, w * 0.045) ** 2))
        vertical_focus = np.exp(-((grid_y - h * 0.48) ** 2) / (2.0 * max(4.0, h * 0.14) ** 2))
        gap_focus = split_focus * vertical_focus * center_mask
        delta_x -= base_shift * 2.20 * gap_focus * left_mask
        delta_x += base_shift * 2.20 * gap_focus * right_mask
        delta_y -= base_shift * 0.18 * gap_focus * upper_mask
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
    final_output: np.ndarray,
    landmarks_debug: np.ndarray,
    blend_mask: np.ndarray,
    teeth_mask: np.ndarray,
    teeth_debug: np.ndarray,
    geometry_debug: np.ndarray,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "mouth_roi.png"), mouth_roi)
    cv2.imwrite(str(output_dir / "warped_roi.png"), warped_roi)
    cv2.imwrite(str(output_dir / "final_output.png"), final_output)
    cv2.imwrite(str(output_dir / "landmarks_debug.png"), landmarks_debug)
    cv2.imwrite(str(output_dir / "blend_mask.png"), blend_mask)
    cv2.imwrite(str(output_dir / "teeth_mask.png"), teeth_mask)
    cv2.imwrite(str(output_dir / "teeth_debug.png"), teeth_debug)
    cv2.imwrite(str(output_dir / "geometry_debug.png"), geometry_debug)

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

    landmarks, detector_mode = detect_landmarks(image, args.face_landmarker_model)
    mouth_roi, bbox = extract_mouth_roi(image, landmarks, margin=args.margin)
    teeth_mask, teeth_debug = build_teeth_mask(mouth_roi, landmarks, bbox)
    teeth_geometry, geometry_debug = build_teeth_geometry(teeth_mask)
    delta_x, delta_y = build_template_delta(
        mouth_roi.shape,
        args.disease,
        severity_value,
        teeth_mask=teeth_mask,
        teeth_geometry=teeth_geometry,
    )
    warped_roi = warp_mouth(mouth_roi, delta_x, delta_y)
    final_output, blend_mask = blend_back(image, warped_roi, bbox, teeth_mask=teeth_mask)
    landmarks_debug = draw_landmarks_debug(image, landmarks, bbox)

    summary = {
        "input": str(Path(args.input).resolve()),
        "detector_mode": detector_mode,
        "disease": args.disease,
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
    }

    save_outputs(
        output_dir=output_dir,
        mouth_roi=mouth_roi,
        warped_roi=warped_roi,
        final_output=final_output,
        landmarks_debug=landmarks_debug,
        blend_mask=blend_mask,
        teeth_mask=teeth_mask,
        teeth_debug=teeth_debug,
        geometry_debug=geometry_debug,
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
