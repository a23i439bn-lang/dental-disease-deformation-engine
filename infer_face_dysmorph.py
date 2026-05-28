from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import cv2
import numpy as np

from disease_templates import SUPPORTED_DISEASE_CHOICES, get_template
from utils.face_landmarks import (
    DEFAULT_FACE_LANDMARKER_PATH,
    detect_landmarks_3d,
    load_image_unicode_safe,
    parse_severity_values,
    save_image_unicode_safe,
)
from utils.face_refine import (
    DEFAULT_SD_MODEL_ID,
    DEFAULT_SD_NEGATIVE_PROMPT,
    refine_face_with_sd_local_edit,
)
from utils.face_shading import apply_pseudo_3d_simulation, render_pseudo_3d_relief_preview
from utils.face_warp import (
    blend_with_original,
    build_delta_heatmap,
    build_dense_displacement,
    draw_landmark_debug,
    remap_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face dysmorph template inference with lower-face geometric warping.")
    parser.add_argument("--input", required=True, help="Input face image path")
    parser.add_argument(
        "--disease",
        default="mandibular_protrusion",
        help=f"Face dysmorph template name. Canonical templates: {', '.join(SUPPORTED_DISEASE_CHOICES)}",
    )
    parser.add_argument("--severity", type=float, default=0.6, help="Strength in [0, 1.5]")
    parser.add_argument(
        "--severity-sweep",
        default="",
        help="Comma-separated severities, e.g. 0.2,0.5,0.8",
    )
    parser.add_argument(
        "--face-landmarker-model",
        default=str(DEFAULT_FACE_LANDMARKER_PATH),
        help="Path to MediaPipe Face Landmarker .task model",
    )
    parser.add_argument("--output-dir", default="face_dysmorph_outputs", help="Directory for saved outputs")
    parser.add_argument("--warp-sigma", type=float, default=52.0, help="Gaussian sigma for dense warp propagation")
    parser.add_argument("--feather", type=int, default=21, help="Feather size for disease blend mask")
    parser.add_argument("--debug-arrows", action="store_true", help="Draw control-point displacement arrows")
    parser.add_argument("--enable-sd-refine", action="store_true", help="Run Stable Diffusion inpaint refinement on the disease region")
    parser.add_argument("--sd-model-id", default=DEFAULT_SD_MODEL_ID, help="Stable Diffusion model id or local directory")
    parser.add_argument("--sd-device", default="auto", help="Diffusion device: auto/cuda/cpu")
    parser.add_argument("--sd-steps", type=int, default=24, help="Inpaint inference steps")
    parser.add_argument("--sd-guidance-scale", type=float, default=4.5, help="Inpaint guidance scale")
    parser.add_argument("--sd-strength", type=float, default=0.2, help="Inpaint denoise strength")
    parser.add_argument("--sd-seed", type=int, default=1234, help="Inpaint random seed")
    parser.add_argument("--sd-prompt", default="", help="Optional full positive prompt override")
    parser.add_argument("--sd-negative-prompt", default=DEFAULT_SD_NEGATIVE_PROMPT, help="Negative prompt for inpaint")
    parser.add_argument("--sd-render-main", action="store_true", help="Blend original context outside the edit mask before inpaint")
    parser.add_argument("--sd-roi-pad-ratio", type=float, default=0.16, help="Padding ratio around the disease mask crop used for inpaint")
    parser.add_argument(
        "--pseudo3d-strength",
        type=float,
        default=1.0,
        help="Pseudo-3D shading/relief strength. Use 2.0-3.0 for visible research inspection.",
    )
    return parser.parse_args()


def save_outputs(
    output_dir: Path,
    original: np.ndarray,
    warped_full: np.ndarray,
    shaded_warped_full: np.ndarray,
    final_output: np.ndarray,
    mask: np.ndarray,
    weight_map: np.ndarray,
    shading_map: np.ndarray,
    pseudo_depth_map: np.ndarray,
    pseudo_highlight_map: np.ndarray,
    pseudo3d_relief_preview: np.ndarray,
    landmarks_debug: np.ndarray,
    heatmap: np.ndarray,
    condition_crop: np.ndarray | None,
    refined_crop: np.ndarray | None,
    disease_edit_mask: np.ndarray | None,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image_unicode_safe(output_dir / "input.png", original)
    save_image_unicode_safe(output_dir / "warped_full.png", warped_full)
    save_image_unicode_safe(output_dir / "shaded_warped_full.png", shaded_warped_full)
    save_image_unicode_safe(output_dir / "final_output.png", final_output)
    save_image_unicode_safe(output_dir / "lower_face_mask.png", mask)
    save_image_unicode_safe(output_dir / "deformation_weight_map.png", weight_map)
    save_image_unicode_safe(output_dir / "shading_map.png", shading_map)
    save_image_unicode_safe(output_dir / "pseudo_depth_map.png", pseudo_depth_map)
    save_image_unicode_safe(output_dir / "pseudo_highlight_map.png", pseudo_highlight_map)
    save_image_unicode_safe(output_dir / "pseudo3d_relief_preview.png", pseudo3d_relief_preview)
    save_image_unicode_safe(output_dir / "landmarks_debug.png", landmarks_debug)
    save_image_unicode_safe(output_dir / "delta_heatmap.png", heatmap)
    if condition_crop is not None:
        save_image_unicode_safe(output_dir / "condition_crop.png", condition_crop)
    if refined_crop is not None:
        save_image_unicode_safe(output_dir / "refined_crop.png", refined_crop)
    if disease_edit_mask is not None:
        save_image_unicode_safe(output_dir / "disease_edit_mask.png", disease_edit_mask)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def render_severity_strip(output_paths: list[Path], save_path: Path) -> None:
    images: list[np.ndarray] = []
    for path in output_paths:
        image = load_image_unicode_safe(str(path))
        if image is None:
            continue
        label = path.parent.name.replace("severity_", "s=")
        canvas = image.copy()
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        images.append(canvas)
    if images:
        strip = cv2.hconcat(images)
        save_image_unicode_safe(save_path, strip)


def run_single_case(image: np.ndarray, args: argparse.Namespace, severity_value: float, output_dir: Path) -> Path:
    template = get_template(args.disease)
    landmark_result = detect_landmarks_3d(image, args.face_landmarker_model)
    landmarks = landmark_result.landmarks_2d
    landmarks_3d = landmark_result.landmarks_3d
    detector_mode = landmark_result.detector_mode
    template_result = template.build(
        image_shape=image.shape,
        landmarks=landmarks,
        severity=severity_value,
        feather=args.feather,
    )
    deformation_weight_map = template_result.weight_map if template_result.weight_map is not None else template_result.mask

    disp_x, disp_y = build_dense_displacement(
        image_shape=image.shape,
        src_points=template_result.src_points,
        dst_points=template_result.dst_points,
        sigma=args.warp_sigma,
        mask=template_result.mask,
        weight_map=deformation_weight_map,
    )
    warped_full = remap_image(image, disp_x, disp_y)
    shaded_warped_full, shading_map, pseudo_depth_map, pseudo_highlight_map = apply_pseudo_3d_simulation(
        warped_full,
        landmarks,
        template_result.canonical_name,
        severity_value,
        template_result.mask,
        strength=args.pseudo3d_strength,
        landmarks_3d=landmarks_3d,
    )
    pseudo3d_relief_preview = render_pseudo_3d_relief_preview(
        shaded_warped_full,
        pseudo_depth_map,
        template_result.mask,
        strength=args.pseudo3d_strength,
    )
    refined_full = shaded_warped_full
    sd_meta: dict[str, object] = {"enabled": False}
    condition_crop: np.ndarray | None = None
    refined_crop: np.ndarray | None = None
    disease_edit_mask: np.ndarray | None = None
    if args.enable_sd_refine:
        refined_full, sd_meta, condition_crop, refined_crop, disease_edit_mask = refine_face_with_sd_local_edit(
            original_image=image,
            geometric_image=shaded_warped_full,
            landmarks=landmarks,
            disease_mask=template_result.mask,
            disease=template_result.canonical_name,
            severity=severity_value,
            args=args,
        )
    final_output = blend_with_original(image, refined_full, template_result.mask)
    landmarks_debug = draw_landmark_debug(
        image,
        landmarks,
        template_result.src_points,
        template_result.dst_points,
        args.debug_arrows,
    )
    heatmap = build_delta_heatmap(disp_x, disp_y)
    z_values = landmarks_3d[:, 2].astype(np.float32)
    chin_z = z_values[[idx for idx in [152, 377, 400, 378, 379] if idx < len(z_values)]]

    summary = {
        "input": str(Path(args.input).resolve()),
        "detector_mode": detector_mode,
        "disease": template_result.canonical_name,
        "requested_disease": args.disease,
        "severity": float(severity_value),
        "warp_sigma": float(args.warp_sigma),
        "feather": int(args.feather),
        "control_points": int(template_result.src_points.shape[0]),
        "mask_pixels": int(cv2.countNonZero(template_result.mask)),
        "weight_map_mean": float(np.mean(deformation_weight_map) / 255.0),
        "weight_map_max": float(np.max(deformation_weight_map) / 255.0),
        "shading_map_mean": float(np.mean(shading_map) / 255.0),
        "shading_map_max": float(np.max(shading_map) / 255.0),
        "pseudo_depth_mean": float(np.mean(pseudo_depth_map) / 255.0),
        "pseudo_depth_max": float(np.max(pseudo_depth_map) / 255.0),
        "pseudo_highlight_mean": float(np.mean(pseudo_highlight_map) / 255.0),
        "pseudo_highlight_max": float(np.max(pseudo_highlight_map) / 255.0),
        "pseudo3d_strength": float(args.pseudo3d_strength),
        "mediapipe_z_min": float(np.min(z_values)),
        "mediapipe_z_max": float(np.max(z_values)),
        "mediapipe_z_range": float(np.max(z_values) - np.min(z_values)),
        "mediapipe_chin_z_mean": float(np.mean(chin_z)) if chin_z.size else 0.0,
        "delta_abs_mean": float(np.mean(np.abs(disp_x)) + np.mean(np.abs(disp_y))),
        "delta_abs_max": float(max(np.max(np.abs(disp_x)), np.max(np.abs(disp_y)))),
        "sd_refine": sd_meta,
        "output_image": str((output_dir / "final_output.png").resolve()),
    }

    save_outputs(
        output_dir=output_dir,
        original=image,
        warped_full=warped_full,
        shaded_warped_full=shaded_warped_full,
        final_output=final_output,
        mask=template_result.mask,
        weight_map=deformation_weight_map,
        shading_map=shading_map,
        pseudo_depth_map=pseudo_depth_map,
        pseudo_highlight_map=pseudo_highlight_map,
        pseudo3d_relief_preview=pseudo3d_relief_preview,
        landmarks_debug=landmarks_debug,
        heatmap=heatmap,
        condition_crop=condition_crop,
        refined_crop=refined_crop,
        disease_edit_mask=disease_edit_mask,
        summary=summary,
    )
    return output_dir / "final_output.png"


def main() -> None:
    args = parse_args()
    image = load_image_unicode_safe(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    output_root = Path(args.output_dir).resolve()
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
