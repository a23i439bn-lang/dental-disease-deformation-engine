from __future__ import annotations

import cv2
import numpy as np

from utils.face_regions import (
    CHIN_IDX,
    LEFT_CHIN_SIDE_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MOUTH_CORNER_IDX,
    MOUTH_OUTER_IDX,
    NOSE_BASE_IDX,
    RIGHT_CHIN_SIDE_IDX,
    RIGHT_LOWER_JAW_IDX,
    UPPER_LIP_IDX,
    face_metrics,
)


def _blur(values: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(values, (0, 0), sigmaX=max(1.0, sigma), sigmaY=max(1.0, sigma))


def _add_polygon(
    shading: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
    strength: float,
    feather: float,
) -> None:
    if not indices or strength <= 0.0:
        return
    region = np.zeros_like(shading, dtype=np.float32)
    points = landmarks[indices].astype(np.int32)
    if len(points) >= 3:
        cv2.fillConvexPoly(region, cv2.convexHull(points), 1.0)
    else:
        for x, y in points:
            cv2.circle(region, (int(x), int(y)), int(max(3.0, feather * 0.35)), 1.0, -1)
    shading[:] = np.maximum(shading, _blur(region, feather) * float(strength))


def _add_line(
    shading: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    strength: float,
    thickness: int,
    feather: float,
) -> None:
    if strength <= 0.0:
        return
    region = np.zeros_like(shading, dtype=np.float32)
    cv2.line(
        region,
        tuple(start.astype(np.int32)),
        tuple(end.astype(np.int32)),
        1.0,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
    )
    shading[:] = np.maximum(shading, _blur(region, feather) * float(strength))


def _offset_point(point: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return point.astype(np.float32) + np.array([dx, dy], dtype=np.float32)


def _normalized_forward_depth(landmarks_3d: np.ndarray) -> np.ndarray:
    z = landmarks_3d[:, 2].astype(np.float32)
    forward = -z
    center = float(np.median(forward))
    spread = float(np.percentile(forward, 95) - np.percentile(forward, 5))
    if spread <= 1e-6:
        return np.zeros_like(forward, dtype=np.float32)
    return np.clip((forward - center) / spread + 0.5, 0.0, 1.0)


def _add_mediapipe_depth_prior(
    depth: np.ndarray,
    landmarks: np.ndarray,
    landmarks_3d: np.ndarray | None,
    groups: list[tuple[list[int], float]],
    feather: float,
) -> None:
    if landmarks_3d is None or landmarks_3d.shape[0] < landmarks.shape[0]:
        return
    forward = _normalized_forward_depth(landmarks_3d)
    for indices, strength in groups:
        if not indices or strength <= 0.0:
            continue
        group_forward = float(np.mean(forward[indices]))
        # Keep the MediaPipe z prior as a local anatomical cue, not a full replacement
        # for the disease-specific depth design.
        _add_polygon(depth, landmarks, indices, group_forward * strength, feather)


def build_disease_shading_map(
    image_shape: tuple[int, int, int],
    landmarks: np.ndarray,
    disease: str,
    severity: float,
    disease_mask: np.ndarray,
) -> np.ndarray:
    height, width = image_shape[:2]
    shading = np.zeros((height, width), dtype=np.float32)
    metrics = face_metrics(landmarks)
    severity = float(np.clip(severity, 0.0, 1.5))
    gain = 0.45 + 0.55 * min(severity, 1.0)
    feather = max(7.0, min(width, height) * 0.018)

    mouth_center = np.array([metrics["mouth_center_x"], metrics["mouth_center_y"]], dtype=np.float32)
    chin_center = landmarks[CHIN_IDX].astype(np.float32).mean(axis=0)
    lower_lip_center = landmarks[LOWER_LIP_IDX].astype(np.float32).mean(axis=0)
    nose_base_center = landmarks[NOSE_BASE_IDX].astype(np.float32).mean(axis=0)
    mouth_width = max(metrics["mouth_width"], 1.0)
    face_height = max(metrics["face_height"], 1.0)

    if disease == "mandibular_protrusion":
        _add_polygon(shading, landmarks, CHIN_IDX, 0.50 * gain, feather * 1.2)
        _add_polygon(shading, landmarks, LOWER_LIP_IDX, 0.36 * gain, feather)
        _add_line(
            shading,
            _offset_point(lower_lip_center, -mouth_width * 0.28, face_height * 0.028),
            _offset_point(lower_lip_center, mouth_width * 0.28, face_height * 0.028),
            0.50 * gain,
            int(max(4, mouth_width * 0.035)),
            feather * 0.75,
        )
        _add_line(
            shading,
            landmarks[LEFT_LOWER_JAW_IDX[-1]].astype(np.float32),
            landmarks[RIGHT_LOWER_JAW_IDX[-1]].astype(np.float32),
            0.22 * gain,
            int(max(3, mouth_width * 0.025)),
            feather * 1.3,
        )
    elif disease == "maxillary_protrusion":
        _add_polygon(shading, landmarks, UPPER_LIP_IDX, 0.34 * gain, feather)
        _add_line(
            shading,
            _offset_point(nose_base_center, -mouth_width * 0.20, face_height * 0.015),
            _offset_point(mouth_center, -mouth_width * 0.36, -face_height * 0.010),
            0.38 * gain,
            int(max(3, mouth_width * 0.024)),
            feather * 0.9,
        )
        _add_line(
            shading,
            _offset_point(nose_base_center, mouth_width * 0.20, face_height * 0.015),
            _offset_point(mouth_center, mouth_width * 0.36, -face_height * 0.010),
            0.38 * gain,
            int(max(3, mouth_width * 0.024)),
            feather * 0.9,
        )
    elif disease.startswith("chin_deviation"):
        deviates_left = disease.endswith("_left")
        strong_side = LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX if deviates_left else RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX
        weak_side = RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX if deviates_left else LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX
        _add_polygon(shading, landmarks, CHIN_IDX, 0.42 * gain, feather * 1.1)
        _add_polygon(shading, landmarks, strong_side, 0.38 * gain, feather * 1.2)
        _add_polygon(shading, landmarks, weak_side, 0.18 * gain, feather * 1.4)
        shift = -mouth_width * 0.25 if deviates_left else mouth_width * 0.25
        _add_line(
            shading,
            _offset_point(chin_center, shift, -face_height * 0.010),
            _offset_point(chin_center, shift * 0.55, face_height * 0.12),
            0.34 * gain,
            int(max(3, mouth_width * 0.026)),
            feather,
        )
    elif disease.startswith("occlusal_plane_cant"):
        left_corner, right_corner = landmarks[MOUTH_CORNER_IDX].astype(np.float32)
        direction = 1.0 if disease.endswith("_right") else -1.0
        low_corner = right_corner if direction > 0.0 else left_corner
        _add_polygon(shading, landmarks, MOUTH_OUTER_IDX, 0.22 * gain, feather * 0.8)
        _add_line(
            shading,
            _offset_point(low_corner, -direction * mouth_width * 0.12, face_height * 0.018),
            _offset_point(low_corner, direction * mouth_width * 0.12, face_height * 0.055),
            0.46 * gain,
            int(max(3, mouth_width * 0.030)),
            feather * 0.75,
        )
        _add_polygon(shading, landmarks, LOWER_LIP_IDX, 0.26 * gain, feather)

    mask = np.clip(disease_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    shading = np.clip(shading * mask, 0.0, 1.0)
    return np.clip(shading * 255.0, 0.0, 255.0).astype(np.uint8)


def build_pseudo_depth_map(
    image_shape: tuple[int, int, int],
    landmarks: np.ndarray,
    disease: str,
    severity: float,
    disease_mask: np.ndarray,
    landmarks_3d: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image_shape[:2]
    depth = np.zeros((height, width), dtype=np.float32)
    metrics = face_metrics(landmarks)
    severity = float(np.clip(severity, 0.0, 1.5))
    gain = 0.40 + 0.60 * min(severity, 1.0)
    feather = max(8.0, min(width, height) * 0.022)

    mouth_width = max(metrics["mouth_width"], 1.0)
    face_height = max(metrics["face_height"], 1.0)
    mouth_center = np.array([metrics["mouth_center_x"], metrics["mouth_center_y"]], dtype=np.float32)
    chin_center = landmarks[CHIN_IDX].astype(np.float32).mean(axis=0)
    upper_lip_center = landmarks[UPPER_LIP_IDX].astype(np.float32).mean(axis=0)

    if disease == "mandibular_protrusion":
        _add_mediapipe_depth_prior(
            depth,
            landmarks,
            landmarks_3d,
            [
                (CHIN_IDX, 0.36 * gain),
                (LOWER_LIP_IDX, 0.26 * gain),
                (LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX, 0.18 * gain),
            ],
            feather,
        )
        _add_polygon(depth, landmarks, CHIN_IDX, 0.78 * gain, feather * 1.25)
        _add_polygon(depth, landmarks, LOWER_LIP_IDX, 0.48 * gain, feather)
        _add_line(
            depth,
            _offset_point(chin_center, -mouth_width * 0.34, -face_height * 0.010),
            _offset_point(chin_center, mouth_width * 0.34, -face_height * 0.010),
            0.38 * gain,
            int(max(4, mouth_width * 0.045)),
            feather,
        )
    elif disease == "maxillary_protrusion":
        _add_mediapipe_depth_prior(
            depth,
            landmarks,
            landmarks_3d,
            [
                (UPPER_LIP_IDX, 0.40 * gain),
                (NOSE_BASE_IDX, 0.24 * gain),
            ],
            feather,
        )
        _add_polygon(depth, landmarks, UPPER_LIP_IDX, 0.70 * gain, feather * 1.05)
        _add_polygon(depth, landmarks, NOSE_BASE_IDX, 0.34 * gain, feather * 1.2)
        _add_line(
            depth,
            _offset_point(upper_lip_center, -mouth_width * 0.30, -face_height * 0.020),
            _offset_point(upper_lip_center, mouth_width * 0.30, -face_height * 0.020),
            0.35 * gain,
            int(max(4, mouth_width * 0.040)),
            feather,
        )
    elif disease.startswith("chin_deviation"):
        deviates_left = disease.endswith("_left")
        strong_side = LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX if deviates_left else RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX
        weak_side = RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX if deviates_left else LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX
        _add_mediapipe_depth_prior(
            depth,
            landmarks,
            landmarks_3d,
            [
                (strong_side, 0.34 * gain),
                (weak_side, 0.16 * gain),
                (CHIN_IDX, 0.28 * gain),
            ],
            feather,
        )
        _add_polygon(depth, landmarks, CHIN_IDX, 0.62 * gain, feather)
        _add_polygon(depth, landmarks, strong_side, 0.46 * gain, feather * 1.25)
        _add_polygon(depth, landmarks, weak_side, 0.20 * gain, feather * 1.35)
    elif disease.startswith("occlusal_plane_cant"):
        left_corner, right_corner = landmarks[MOUTH_CORNER_IDX].astype(np.float32)
        direction = 1.0 if disease.endswith("_right") else -1.0
        high_corner = left_corner if direction > 0.0 else right_corner
        _add_mediapipe_depth_prior(
            depth,
            landmarks,
            landmarks_3d,
            [
                (MOUTH_OUTER_IDX, 0.28 * gain),
                (LOWER_LIP_IDX, 0.20 * gain),
            ],
            feather,
        )
        _add_polygon(depth, landmarks, MOUTH_OUTER_IDX, 0.42 * gain, feather)
        _add_line(
            depth,
            _offset_point(mouth_center, -mouth_width * 0.36, 0.0),
            _offset_point(mouth_center, mouth_width * 0.36, 0.0),
            0.28 * gain,
            int(max(4, mouth_width * 0.035)),
            feather,
        )
        _add_line(
            depth,
            _offset_point(high_corner, -direction * mouth_width * 0.18, -face_height * 0.020),
            _offset_point(high_corner, direction * mouth_width * 0.10, -face_height * 0.010),
            0.36 * gain,
            int(max(3, mouth_width * 0.030)),
            feather * 0.8,
        )

    mask = np.clip(disease_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    depth = np.clip(depth * mask, 0.0, 1.0)
    return np.clip(depth * 255.0, 0.0, 255.0).astype(np.uint8)


def build_pseudo_highlight_map(
    depth_map: np.ndarray,
    shading_map: np.ndarray,
    disease_mask: np.ndarray,
) -> np.ndarray:
    depth = np.clip(depth_map.astype(np.float32) / 255.0, 0.0, 1.0)
    shadow = np.clip(shading_map.astype(np.float32) / 255.0, 0.0, 1.0)
    mask = np.clip(disease_mask.astype(np.float32) / 255.0, 0.0, 1.0)

    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    light_from_upper_left = np.clip(-grad_x * 0.50 - grad_y * 0.72, 0.0, 1.0)
    convex_lift = np.clip(depth - shadow * 0.34, 0.0, 1.0)
    highlight = np.clip(light_from_upper_left * 0.55 + convex_lift * 0.30, 0.0, 1.0)
    highlight = _blur(highlight * mask, max(3.0, min(depth_map.shape[:2]) * 0.010))
    return np.clip(highlight * 255.0, 0.0, 255.0).astype(np.uint8)


def apply_pseudo_3d_simulation(
    image: np.ndarray,
    landmarks: np.ndarray,
    disease: str,
    severity: float,
    disease_mask: np.ndarray,
    strength: float = 1.0,
    landmarks_3d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shading_map = build_disease_shading_map(image.shape, landmarks, disease, severity, disease_mask)
    depth_map = build_pseudo_depth_map(image.shape, landmarks, disease, severity, disease_mask, landmarks_3d=landmarks_3d)
    highlight_map = build_pseudo_highlight_map(depth_map, shading_map, disease_mask)

    strength = float(np.clip(strength, 0.0, 4.0))
    shadow = np.clip(shading_map.astype(np.float32) / 255.0 * strength, 0.0, 1.0)
    highlight = np.clip(highlight_map.astype(np.float32) / 255.0 * strength, 0.0, 1.0)
    if float(max(shadow.max(), highlight.max())) <= 1e-6:
        return image.copy(), shading_map, depth_map, highlight_map

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] - shadow * 32.0 + highlight * 18.0, 0.0, 255.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + shadow * 1.6 - highlight * 0.8, 0.0, 255.0)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + highlight * 1.2, 0.0, 255.0)
    shaded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    effect_alpha = np.clip((shadow + highlight)[:, :, None] * 0.78, 0.0, 0.82)
    out = shaded.astype(np.float32) * effect_alpha + image.astype(np.float32) * (1.0 - effect_alpha)
    return np.clip(out, 0, 255).astype(np.uint8), shading_map, depth_map, highlight_map


def render_pseudo_3d_relief_preview(
    image: np.ndarray,
    depth_map: np.ndarray,
    disease_mask: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = np.clip(depth_map.astype(np.float32) / 255.0, 0.0, 1.0)
    mask = np.clip(disease_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    depth = _blur(depth * mask, max(2.0, min(width, height) * 0.006))
    max_shift = max(4.0, width * 0.035) * float(np.clip(strength, 0.5, 4.0))

    views: list[np.ndarray] = []
    for yaw, label in [(-1.0, "left view"), (0.0, "front"), (1.0, "right view")]:
        map_x = (xx - yaw * depth * max_shift).astype(np.float32)
        map_y = (yy + np.abs(yaw) * depth * max_shift * 0.12).astype(np.float32)
        view = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        alpha = np.clip(mask[:, :, None] * 0.96, 0.0, 1.0)
        view = np.clip(view.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
        cv2.putText(view, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(view, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        views.append(view)

    return cv2.hconcat(views)


def apply_shading_simulation(
    image: np.ndarray,
    landmarks: np.ndarray,
    disease: str,
    severity: float,
    disease_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shaded, shading_map, _, _ = apply_pseudo_3d_simulation(image, landmarks, disease, severity, disease_mask)
    return shaded, shading_map
