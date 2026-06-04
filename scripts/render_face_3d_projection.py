from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from disease_templates import get_template
from utils.face_landmarks import DEFAULT_FACE_LANDMARKER_PATH, detect_landmarks_3d, load_image_unicode_safe, save_image_unicode_safe
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a MediaPipe 3D-landmark disease transform from virtual camera views.")
    parser.add_argument("--input", required=True, help="Input face image path")
    parser.add_argument("--disease", required=True, help="Disease template name")
    parser.add_argument("--severity", type=float, default=1.0, help="Disease severity")
    parser.add_argument("--output-dir", default="outputs/mediapipe3d_projection", help="Output directory")
    parser.add_argument("--face-landmarker-model", default=str(DEFAULT_FACE_LANDMARKER_PATH))
    parser.add_argument("--feather", type=int, default=21)
    parser.add_argument("--focal-scale", type=float, default=2.4, help="Virtual camera focal length as image-width multiplier")
    parser.add_argument("--z-scale", type=float, default=1.45, help="Amplifies MediaPipe z before projection")
    parser.add_argument("--projection-scale", type=float, default=1.0, help="Final render scale around face center")
    return parser.parse_args()


def delaunay_triangles(image_shape: tuple[int, int, int], landmarks_2d: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = image_shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in landmarks_2d.astype(np.float32):
        if 0 <= x < w and 0 <= y < h:
            subdiv.insert((float(x), float(y)))

    points = landmarks_2d.astype(np.float32)
    triangles: set[tuple[int, int, int]] = set()
    for tri in subdiv.getTriangleList():
        coords = tri.reshape(3, 2).astype(np.float32)
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= w) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= h):
            continue
        indices: list[int] = []
        for coord in coords:
            distances = np.sum((points - coord) ** 2, axis=1)
            indices.append(int(np.argmin(distances)))
        if len(set(indices)) == 3:
            triangles.add(tuple(sorted(indices)))
    return sorted(triangles)


def nearest_landmark_indices(landmarks_2d: np.ndarray, src_points: np.ndarray) -> list[int]:
    points = landmarks_2d.astype(np.float32)
    indices: list[int] = []
    for src in src_points.astype(np.float32):
        distances = np.sum((points - src) ** 2, axis=1)
        indices.append(int(np.argmin(distances)))
    return indices


def apply_template_xy(landmarks_2d: np.ndarray, landmarks_3d: np.ndarray, disease: str, severity: float, feather: int) -> np.ndarray:
    template = get_template(disease)
    # Only image shape size matters for masks inside templates.
    h = int(max(landmarks_2d[:, 1].max() + 32, 256))
    w = int(max(landmarks_2d[:, 0].max() + 32, 256))
    result = template.build((h, w, 3), landmarks_2d, severity, feather)
    target = landmarks_3d.astype(np.float32).copy()
    for idx, dst in zip(nearest_landmark_indices(landmarks_2d, result.src_points), result.dst_points.astype(np.float32)):
        target[idx, 0] = dst[0]
        target[idx, 1] = dst[1]
    return target


def add_forward(target: np.ndarray, indices: list[int], amount: float) -> None:
    valid = [idx for idx in indices if idx < target.shape[0]]
    if valid:
        # MediaPipe z is negative toward the camera, so subtracting makes the
        # region project forward in the virtual view.
        target[valid, 2] -= float(amount)


def add_backward(target: np.ndarray, indices: list[int], amount: float) -> None:
    valid = [idx for idx in indices if idx < target.shape[0]]
    if valid:
        target[valid, 2] += float(amount)


def apply_disease_z_transform(target: np.ndarray, landmarks_2d: np.ndarray, disease: str, severity: float) -> np.ndarray:
    out = target.astype(np.float32).copy()
    metrics = face_metrics(landmarks_2d)
    fw = max(metrics["face_width"], 1.0)
    sev = float(np.clip(severity, 0.0, 1.5))

    if disease == "mandibular_protrusion":
        add_forward(out, CHIN_IDX, fw * (0.060 + 0.095 * sev))
        add_forward(out, LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX, fw * (0.032 + 0.058 * sev))
        add_forward(out, LOWER_LIP_IDX, fw * (0.032 + 0.052 * sev))
        add_forward(out, [17, 18, 200, 199, 175, 152], fw * (0.038 + 0.062 * sev))
    elif disease == "maxillary_protrusion":
        add_forward(out, UPPER_LIP_IDX, fw * (0.046 + 0.078 * sev))
        add_forward(out, MOUTH_OUTER_IDX, fw * (0.026 + 0.048 * sev))
        add_forward(out, NOSE_BASE_IDX, fw * (0.014 + 0.026 * sev))
        add_backward(out, CHIN_IDX, fw * (0.022 + 0.044 * sev))
        add_backward(out, LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX, fw * (0.008 + 0.020 * sev))
    elif disease.startswith("chin_deviation"):
        deviates_left = disease.endswith("_left")
        strong = LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX if deviates_left else RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX
        weak = RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX if deviates_left else LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX
        add_forward(out, strong + CHIN_IDX, fw * (0.030 + 0.052 * sev))
        add_backward(out, weak, fw * (0.012 + 0.028 * sev))
    elif disease.startswith("occlusal_plane_cant"):
        left_corner, right_corner = MOUTH_CORNER_IDX
        if disease.endswith("_right"):
            add_forward(out, [right_corner] + RIGHT_LOWER_JAW_IDX, fw * (0.022 + 0.040 * sev))
            add_backward(out, [left_corner] + LEFT_LOWER_JAW_IDX, fw * (0.010 + 0.020 * sev))
        else:
            add_forward(out, [left_corner] + LEFT_LOWER_JAW_IDX, fw * (0.022 + 0.040 * sev))
            add_backward(out, [right_corner] + RIGHT_LOWER_JAW_IDX, fw * (0.010 + 0.020 * sev))
        add_forward(out, MOUTH_OUTER_IDX, fw * (0.010 + 0.018 * sev))
    return out


def project_points(points_3d: np.ndarray, image_shape: tuple[int, int, int], yaw_deg: float, focal_scale: float, z_scale: float, projection_scale: float) -> np.ndarray:
    h, w = image_shape[:2]
    center_xy = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    points = points_3d.astype(np.float32).copy()
    center = np.median(points, axis=0)
    local = points - center
    local[:, 2] *= float(z_scale)

    yaw = np.deg2rad(float(yaw_deg))
    cos_y = float(np.cos(yaw))
    sin_y = float(np.sin(yaw))
    x = local[:, 0] * cos_y + local[:, 2] * sin_y
    z = -local[:, 0] * sin_y + local[:, 2] * cos_y
    y = local[:, 1]

    focal = max(w, h) * float(focal_scale)
    scale = focal / np.clip(focal + z, focal * 0.25, focal * 4.0)
    projected = np.zeros((points.shape[0], 2), dtype=np.float32)
    projected[:, 0] = x * scale * projection_scale + center_xy[0]
    projected[:, 1] = y * scale * projection_scale + center_xy[1]
    return projected


def camera_z_values(points_3d: np.ndarray, yaw_deg: float, z_scale: float) -> np.ndarray:
    points = points_3d.astype(np.float32).copy()
    center = np.median(points, axis=0)
    local = points - center
    local[:, 2] *= float(z_scale)
    yaw = np.deg2rad(float(yaw_deg))
    sin_y = float(np.sin(yaw))
    cos_y = float(np.cos(yaw))
    return -local[:, 0] * sin_y + local[:, 2] * cos_y


def affine_warp_triangle(src: np.ndarray, dst: np.ndarray, image: np.ndarray, canvas: np.ndarray, alpha: np.ndarray) -> None:
    src_edges = np.array(
        [
            np.linalg.norm(src[0] - src[1]),
            np.linalg.norm(src[1] - src[2]),
            np.linalg.norm(src[2] - src[0]),
        ],
        dtype=np.float32,
    )
    dst_edges = np.array(
        [
            np.linalg.norm(dst[0] - dst[1]),
            np.linalg.norm(dst[1] - dst[2]),
            np.linalg.norm(dst[2] - dst[0]),
        ],
        dtype=np.float32,
    )
    src_area = max(abs(cv2.contourArea(src.astype(np.float32))), 1.0)
    dst_area = max(abs(cv2.contourArea(dst.astype(np.float32))), 1.0)
    if float(dst_edges.max() / max(src_edges.max(), 1.0)) > 2.6:
        return
    area_ratio = dst_area / src_area
    if area_ratio > 4.5 or area_ratio < 0.12:
        return

    src_rect = cv2.boundingRect(src.astype(np.float32))
    dst_rect = cv2.boundingRect(dst.astype(np.float32))
    x, y, w, h = dst_rect
    if w <= 1 or h <= 1:
        return
    ih, iw = image.shape[:2]
    if x >= iw or y >= ih or x + w <= 0 or y + h <= 0:
        return

    src_offset = src - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
    dst_offset = dst - np.array([x, y], dtype=np.float32)
    sx, sy, sw, sh = src_rect
    if sx < 0 or sy < 0 or sx + sw > iw or sy + sh > ih or sw <= 1 or sh <= 1:
        return

    patch = image[sy:sy + sh, sx:sx + sw]
    transform = cv2.getAffineTransform(src_offset.astype(np.float32), dst_offset.astype(np.float32))
    warped = cv2.warpAffine(patch, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    tri_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(tri_mask, np.round(dst_offset).astype(np.int32), 1.0, lineType=cv2.LINE_AA)

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, iw)
    y1 = min(y + h, ih)
    if x1 <= x0 or y1 <= y0:
        return
    px0 = x0 - x
    py0 = y0 - y
    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)
    roi_mask = tri_mask[py0:py1, px0:px1, None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - roi_mask) + warped[py0:py1, px0:px1].astype(np.float32) * roi_mask
    alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], tri_mask[py0:py1, px0:px1])


def render_mesh_projection(
    image: np.ndarray,
    source_2d: np.ndarray,
    target_2d: np.ndarray,
    triangles: list[tuple[int, int, int]],
    point_camera_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    canvas = cv2.GaussianBlur(image, (0, 0), sigmaX=4.0, sigmaY=4.0).astype(np.float32)
    alpha = np.zeros(image.shape[:2], dtype=np.float32)
    h, w = image.shape[:2]
    draw_tris = triangles
    if point_camera_z is not None:
        draw_tris = sorted(triangles, key=lambda tri: float(np.mean(point_camera_z[list(tri)])), reverse=True)
    for tri in draw_tris:
        src = source_2d[list(tri)].astype(np.float32)
        dst = target_2d[list(tri)].astype(np.float32)
        if np.any(dst[:, 0] < -w * 0.4) or np.any(dst[:, 0] > w * 1.4) or np.any(dst[:, 1] < -h * 0.4) or np.any(dst[:, 1] > h * 1.4):
            continue
        if abs(cv2.contourArea(dst.astype(np.float32))) < 1.0:
            continue
        affine_warp_triangle(src, dst, image, canvas, alpha)

    alpha_blur = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha_blur = np.clip(alpha_blur[:, :, None], 0.0, 1.0)
    out = canvas * alpha_blur + image.astype(np.float32) * (1.0 - alpha_blur)
    return np.clip(out, 0, 255).astype(np.uint8), np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    image = load_image_unicode_safe(args.input)
    if image is None:
        raise FileNotFoundError(args.input)

    landmark_result = detect_landmarks_3d(image, args.face_landmarker_model)
    landmarks_2d = landmark_result.landmarks_2d
    normal_3d = landmark_result.landmarks_3d
    target_3d = apply_template_xy(landmarks_2d, normal_3d, args.disease, args.severity, args.feather)
    target_3d = apply_disease_z_transform(target_3d, landmarks_2d, args.disease, args.severity)
    triangles = delaunay_triangles(image.shape, landmarks_2d)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_images: list[np.ndarray] = [label_image(image, "input")]
    render_manifest: dict[str, str] = {}
    for view_name, yaw in [("front", 0.0), ("oblique_left", -22.0), ("oblique_right", 22.0), ("side_left", -48.0), ("side_right", 48.0)]:
        projected = project_points(
            target_3d,
            image.shape,
            yaw_deg=yaw,
            focal_scale=args.focal_scale,
            z_scale=args.z_scale,
            projection_scale=args.projection_scale,
        )
        point_z = camera_z_values(target_3d, yaw, args.z_scale)
        rendered, mask = render_mesh_projection(image, landmarks_2d.astype(np.float32), projected, triangles, point_camera_z=point_z)
        render_path = out_dir / f"{view_name}.png"
        mask_path = out_dir / f"{view_name}_mesh_mask.png"
        save_image_unicode_safe(render_path, rendered)
        save_image_unicode_safe(mask_path, mask)
        render_manifest[view_name] = str(render_path.resolve())
        panel_images.append(label_image(rendered, view_name.replace("_", " ")))

    panel = cv2.hconcat(panel_images)
    panel_path = out_dir / "projection_panel.png"
    save_image_unicode_safe(panel_path, panel)
    np.save(out_dir / "normal_landmarks_3d.npy", normal_3d)
    np.save(out_dir / "disease_landmarks_3d.npy", target_3d)

    summary = {
        "input": str(Path(args.input).resolve()),
        "detector_mode": landmark_result.detector_mode,
        "disease": get_template(args.disease).canonical_name,
        "severity": float(args.severity),
        "focal_scale": float(args.focal_scale),
        "z_scale": float(args.z_scale),
        "projection_scale": float(args.projection_scale),
        "num_triangles": len(triangles),
        "panel": str(panel_path.resolve()),
        "renders": render_manifest,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
