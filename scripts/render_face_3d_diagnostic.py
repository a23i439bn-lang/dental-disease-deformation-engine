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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disease_templates import get_template
from utils.face_landmarks import (
    DEFAULT_FACE_LANDMARKER_PATH,
    detect_landmarks_3d,
    load_image_unicode_safe,
    save_image_unicode_safe,
)
from utils.face_regions import (
    CHIN_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MOUTH_OUTER_IDX,
    NOSE_BASE_IDX,
    RIGHT_LOWER_JAW_IDX,
    UPPER_LIP_IDX,
)
from utils.face_shading import build_pseudo_depth_map
from utils.face_warp import build_dense_displacement, remap_image


PLOT_GROUPS = [
    ("jaw L", LEFT_LOWER_JAW_IDX, "#2f80ed"),
    ("chin", CHIN_IDX, "#eb5757"),
    ("jaw R", RIGHT_LOWER_JAW_IDX, "#27ae60"),
    ("mouth", MOUTH_OUTER_IDX, "#f2994a"),
    ("upper lip", UPPER_LIP_IDX, "#9b51e0"),
    ("lower lip", LOWER_LIP_IDX, "#56ccf2"),
    ("nose base", NOSE_BASE_IDX, "#828282"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pseudo-3D diagnostic landmark views for face dysmorph templates.")
    parser.add_argument("--input", required=True, help="Input face image path")
    parser.add_argument("--disease", required=True, help="Disease template name")
    parser.add_argument("--severity", type=float, default=1.0, help="Disease severity")
    parser.add_argument("--output-dir", default="outputs/face_3d_diagnostic", help="Output directory")
    parser.add_argument("--face-landmarker-model", default=str(DEFAULT_FACE_LANDMARKER_PATH))
    parser.add_argument("--warp-sigma", type=float, default=52.0)
    parser.add_argument("--feather", type=int, default=21)
    parser.add_argument("--depth-scale", type=float, default=95.0, help="Visual z amplification for diagnostic plotting")
    return parser.parse_args()


def sample_map(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    return values[ys, xs].astype(np.float32)


def make_xyz(points: np.ndarray, z_values: np.ndarray) -> np.ndarray:
    xyz = np.zeros((points.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = points[:, 0]
    xyz[:, 1] = -points[:, 1]
    xyz[:, 2] = z_values
    return xyz


def set_equal_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_view(ax, normal_xyz: np.ndarray, disease_xyz: np.ndarray, title: str, elev: float, azim: float) -> None:
    all_xyz = np.vstack([normal_xyz, disease_xyz])
    for _, indices, color in PLOT_GROUPS:
        normal = normal_xyz[indices]
        disease = disease_xyz[indices]
        ax.plot(normal[:, 0], normal[:, 1], normal[:, 2], color="#bdbdbd", linewidth=1.0, alpha=0.75)
        ax.plot(disease[:, 0], disease[:, 1], disease[:, 2], color=color, linewidth=2.4, alpha=0.95)
        ax.scatter(disease[:, 0], disease[:, 1], disease[:, 2], color=color, s=10, alpha=0.9)

    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("pseudo-depth")
    set_equal_axes(ax, all_xyz)


def render_diagnostic_panel(
    image: np.ndarray,
    warped: np.ndarray,
    normal_xyz: np.ndarray,
    disease_xyz: np.ndarray,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 8), dpi=150)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    for idx, (img, title) in enumerate([(image, "input"), (warped, "warped face")]):
        ax_img = fig.add_subplot(grid[0, idx])
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.set_title(title)
        ax_img.axis("off")

    ax_legend = fig.add_subplot(grid[0, 2])
    ax_legend.axis("off")
    ax_legend.text(0.02, 0.88, "gray: normal landmarks", fontsize=11, color="#555555")
    ax_legend.text(0.02, 0.72, "color: disease template", fontsize=11, color="#222222")
    ax_legend.text(0.02, 0.52, "This is a diagnostic pseudo-3D view.", fontsize=10)
    ax_legend.text(0.02, 0.38, "It makes skeletal block motion visible.", fontsize=10)

    views = [
        ("front", 8, -90),
        ("oblique", 12, -45),
        ("side", 8, 0),
    ]
    for col, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(grid[1, col], projection="3d")
        plot_view(ax, normal_xyz, disease_xyz, title, elev, azim)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    image = load_image_unicode_safe(args.input)
    if image is None:
        raise FileNotFoundError(args.input)

    landmark_result = detect_landmarks_3d(image, args.face_landmarker_model)
    landmarks = landmark_result.landmarks_2d
    landmarks_3d = landmark_result.landmarks_3d
    detector_mode = landmark_result.detector_mode
    template = get_template(args.disease)
    result = template.build(image.shape, landmarks, args.severity, args.feather)
    weight_map = result.weight_map if result.weight_map is not None else result.mask
    disp_x, disp_y = build_dense_displacement(
        image_shape=image.shape,
        src_points=result.src_points,
        dst_points=result.dst_points,
        sigma=args.warp_sigma,
        mask=result.mask,
        weight_map=weight_map,
    )
    warped = remap_image(image, disp_x, disp_y)
    depth_map = build_pseudo_depth_map(
        image.shape,
        landmarks,
        result.canonical_name,
        args.severity,
        result.mask,
        landmarks_3d=landmarks_3d,
    )

    full_dst = landmarks.astype(np.float32).copy()
    for src, dst in zip(result.src_points, result.dst_points):
        matches = np.where(np.all(landmarks.astype(np.float32) == src.astype(np.float32), axis=1))[0]
        if matches.size:
            full_dst[int(matches[0])] = dst

    mp_forward = -landmarks_3d[:, 2].astype(np.float32)
    mp_forward -= float(np.median(mp_forward))
    mp_spread = float(np.percentile(mp_forward, 95) - np.percentile(mp_forward, 5))
    if mp_spread > 1e-6:
        mp_forward = mp_forward / mp_spread * float(args.depth_scale) * 0.62
    normal_z = mp_forward.astype(np.float32)
    disease_z = normal_z + sample_map(depth_map, landmarks) / 255.0 * float(args.depth_scale)
    normal_xyz = make_xyz(landmarks.astype(np.float32), normal_z)
    disease_xyz = make_xyz(full_dst.astype(np.float32), disease_z)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image_unicode_safe(output_dir / "warped_face.png", warped)
    save_image_unicode_safe(output_dir / "pseudo_depth_map.png", depth_map)
    panel_path = output_dir / "pseudo3d_landmark_diagnostic.png"
    render_diagnostic_panel(image, warped, normal_xyz, disease_xyz, panel_path)

    summary = {
        "input": str(Path(args.input).resolve()),
        "detector_mode": detector_mode,
        "disease": result.canonical_name,
        "severity": float(args.severity),
        "depth_scale": float(args.depth_scale),
        "output_panel": str(panel_path.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
