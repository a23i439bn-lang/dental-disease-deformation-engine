from __future__ import annotations

import cv2
import numpy as np

from utils.face_regions import CHIN_IDX, FACE_OVAL_IDX, LOWER_FACE_OVAL_IDX, MOUTH_OUTER_IDX


def build_dense_displacement(
    image_shape: tuple[int, int, int],
    src_points: np.ndarray,
    dst_points: np.ndarray,
    sigma: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    disp_x = np.zeros((height, width), dtype=np.float32)
    disp_y = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    sigma_sq = max(float(sigma) ** 2, 1.0)
    point_delta = dst_points - src_points
    for point, delta in zip(src_points, point_delta):
        dx = xx - float(point[0])
        dy = yy - float(point[1])
        weight = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_sq))
        disp_x += weight * float(delta[0])
        disp_y += weight * float(delta[1])
        weight_sum += weight

    weight_sum = np.clip(weight_sum, 1e-6, None)
    disp_x = disp_x / weight_sum
    disp_y = disp_y / weight_sum

    mask_f = mask.astype(np.float32) / 255.0
    disp_x *= mask_f
    disp_y *= mask_f
    return disp_x, disp_y


def remap_image(image: np.ndarray, disp_x: np.ndarray, disp_y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = xx - disp_x
    map_y = yy - disp_y
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def blend_with_original(image: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    out = warped.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_landmark_debug(
    image: np.ndarray,
    landmarks: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    draw_arrows: bool,
) -> np.ndarray:
    canvas = image.copy()
    cv2.polylines(canvas, [landmarks[FACE_OVAL_IDX].astype(np.int32)], isClosed=True, color=(80, 180, 255), thickness=1)
    cv2.polylines(canvas, [landmarks[MOUTH_OUTER_IDX].astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.polylines(canvas, [landmarks[LOWER_FACE_OVAL_IDX].astype(np.int32)], isClosed=False, color=(255, 180, 0), thickness=2)
    for x, y in landmarks[CHIN_IDX]:
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)
    if draw_arrows:
        for src, dst in zip(src_points.astype(np.int32), dst_points.astype(np.int32)):
            cv2.arrowedLine(canvas, tuple(src), tuple(dst), (255, 0, 255), 1, tipLength=0.2)
    return canvas


def build_delta_heatmap(disp_x: np.ndarray, disp_y: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(disp_x ** 2 + disp_y ** 2)
    if float(magnitude.max()) <= 1e-6:
        normalized = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        normalized = np.clip((magnitude / magnitude.max()) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
