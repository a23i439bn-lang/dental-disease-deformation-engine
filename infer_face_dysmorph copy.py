'''顔の形をテンプレートに合わせて変形し、必要に応じてStable Diffusion（画像生成AI）でキレイに仕上げる実行プログラム'''
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
    detect_landmarks,
    load_image_unicode_safe,
    parse_severity_values,
    save_image_unicode_safe,
)
from utils.face_refine import (
    DEFAULT_SD_MODEL_ID,
    DEFAULT_SD_NEGATIVE_PROMPT,
    refine_face_with_sd_local_edit,
)
from utils.face_warp import (
    blend_with_original,
    build_delta_heatmap,
    build_dense_displacement,
    draw_landmark_debug,
    remap_image,
)

'''あらかじめ用意した型に合わせて顔の形を歪ませる処理と、
お好みで画像生成AI（Stable Diffusion）を使ってキレイに仕上げるための実行プログラム'''
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
    return parser.parse_args()

''' 指定された顔画像に対して、選択された疾患テンプレートに基づいて顔の形を歪ませる処理を実行'''
def save_outputs(
    output_dir: Path,
    original: np.ndarray,
    warped_full: np.ndarray,
    final_output: np.ndarray,
    mask: np.ndarray,
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
    save_image_unicode_safe(output_dir / "final_output.png", final_output)
    save_image_unicode_safe(output_dir / "lower_face_mask.png", mask)
    save_image_unicode_safe(output_dir / "landmarks_debug.png", landmarks_debug)
    save_image_unicode_safe(output_dir / "delta_heatmap.png", heatmap)
    if condition_crop is not None:
        save_image_unicode_safe(output_dir / "condition_crop.png", condition_crop)
    if refined_crop is not None:
        save_image_unicode_safe(output_dir / "refined_crop.png", refined_crop)
    if disease_edit_mask is not None:
        save_image_unicode_safe(output_dir / "disease_edit_mask.png", disease_edit_mask)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

'''複数の歪み強度で処理した結果を横に並べた画像を生成'''
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

'''単一の歪み強度で顔変形処理を実行し、結果を保存する'''
def run_single_case(image: np.ndarray, args: argparse.Namespace, severity_value: float, output_dir: Path) -> Path:
    template = get_template(args.disease)
    landmarks, detector_mode = detect_landmarks(image, args.face_landmarker_model)
    template_result = template.build(
        image_shape=image.shape,
        landmarks=landmarks,
        severity=severity_value,
        feather=args.feather,
    )

    disp_x, disp_y = build_dense_displacement(
        image_shape=image.shape,
        src_points=template_result.src_points,
        dst_points=template_result.dst_points,
        sigma=args.warp_sigma,
        mask=template_result.mask,
    )
    warped_full = remap_image(image, disp_x, disp_y)
    refined_full = warped_full
    sd_meta: dict[str, object] = {"enabled": False}
    condition_crop: np.ndarray | None = None
    refined_crop: np.ndarray | None = None
    disease_edit_mask: np.ndarray | None = None
    if args.enable_sd_refine:
        refined_full, sd_meta, condition_crop, refined_crop, disease_edit_mask = refine_face_with_sd_local_edit(
            original_image=image,
            geometric_image=warped_full,
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
        "delta_abs_mean": float(np.mean(np.abs(disp_x)) + np.mean(np.abs(disp_y))),
        "delta_abs_max": float(max(np.max(np.abs(disp_x)), np.max(np.abs(disp_y)))),
        "sd_refine": sd_meta,
        "output_image": str((output_dir / "final_output.png").resolve()),
    }

    save_outputs(
        output_dir=output_dir,
        original=image,
        warped_full=warped_full,
        final_output=final_output,
        mask=template_result.mask,
        landmarks_debug=landmarks_debug,
        heatmap=heatmap,
        condition_crop=condition_crop,
        refined_crop=refined_crop,
        disease_edit_mask=disease_edit_mask,
        summary=summary,
    )
    return output_dir / "final_output.png"

'''コマンドライン引数を処理し、顔変形処理を実行して結果を保存するメイン関数'''
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
