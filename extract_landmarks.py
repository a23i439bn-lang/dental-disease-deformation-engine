"""Extract MediaPipe Face Landmarker landmarks from a folder of images.

This script is designed for dental facial research workflows.

Features
--------
- Scans all supported images under ``dataset/image``.
- Runs MediaPipe Face Landmarker in IMAGE mode.
- Saves all detected landmarks to ``outputs/landmarks.csv``.
- Saves failed image IDs to ``outputs/failed_images.csv``.
- Uses tqdm progress reporting.
- Continues processing even if an image is corrupted or unreadable.

Default layout
--------------
project/
    dataset/image/
    models/face_landmarker.task
    outputs/
    extract_landmarks.py
"""

from __future__ import annotations

import argparse
import platform
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Keep matplotlib-related cache writes away from a read-only home directory.
os.environ.setdefault("MPLCONFIGDIR", str((BASE_DIR / ".mplconfig").resolve()))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.vision.core import (
    vision_task_running_mode as running_mode_module,
)
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_DIR = BASE_DIR / "dataset" / "image"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
DEFAULT_FALLBACK_MODEL_PATH = BASE_DIR / "face_landmarker.task"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
LANDMARK_COLUMNS = ["image_id", "landmark_id", "x", "y", "z"]
FAILED_COLUMNS = ["image_id"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The defaults match the folder structure described in the request.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe Face Landmarker landmarks from all images in a folder."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to face_landmarker.task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be saved.",
    )
    parser.add_argument(
        "--test-face-landmarker",
        action="store_true",
        help=(
            "Load only the Face Landmarker task file and print detailed "
            "debug information."
        ),
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path) -> Path:
    """Resolve the MediaPipe task model path.

    The requested structure uses ``models/face_landmarker.task``.
    For compatibility with existing repositories, we also fall back to a
    top-level ``face_landmarker.task`` if present.
    """

    candidates = [model_path, DEFAULT_FALLBACK_MODEL_PATH]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidate_list = "\n".join(f"- {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        "Face Landmarker model file was not found.\n"
        "Expected one of the following paths:\n"
        f"{candidate_list}"
    )


def collect_image_paths(image_dir: Path) -> list[Path]:
    """Collect all supported image files under ``image_dir``.

    The scan is recursive so the script can handle nested image folders too.
    """

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")

    image_paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return image_paths


def image_id_from_path(image_path: Path, image_root: Path) -> str:
    """Create a stable image ID from a file path.

    We use the relative path without extension so IDs stay readable and unique
    across nested folders.
    """

    relative_path = image_path.relative_to(image_root)
    return relative_path.with_suffix("").as_posix()


def load_image_bgr(image_path: Path) -> np.ndarray | None:
    """Load an image with OpenCV in a Windows-safe way.

    ``cv2.imread`` can fail on Unicode paths in some Windows setups.
    Using ``np.fromfile`` plus ``cv2.imdecode`` avoids that issue.
    """

    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image_bgr
    except Exception:
        return None


def path_debug_info(model_path: Path) -> dict[str, object]:
    """Collect detailed debug information for the model path."""

    resolved_path = model_path.resolve()
    info: dict[str, object] = {
        "input_path": str(model_path),
        "resolved_path": str(resolved_path),
        "exists": model_path.exists(),
        "is_file": model_path.is_file(),
        "size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "cwd": str(Path.cwd()),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "opencv_version": getattr(cv2, "__version__", "unknown"),
        "platform": platform.platform(),
    }
    return info


def print_debug_info(model_path: Path) -> None:
    """Print environment and path diagnostics."""

    info = path_debug_info(model_path)
    print("\n[DEBUG] Environment information")
    for key in (
        "input_path",
        "resolved_path",
        "exists",
        "is_file",
        "size_bytes",
        "cwd",
        "python_version",
        "python_executable",
        "mediapipe_version",
        "opencv_version",
        "platform",
    ):
        print(f"[DEBUG] {key}: {info[key]}")


def _build_face_landmarker_from_resolved_path(
    resolved_model_path: Path,
) -> face_landmarker.FaceLandmarker:
    """Create a FaceLandmarker configured for IMAGE mode."""

    base_options = base_options_module.BaseOptions(
        model_asset_path=str(resolved_model_path)
    )
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode_module.VisionTaskRunningMode.IMAGE,
        num_faces=1,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


def build_face_landmarker(model_path: Path) -> face_landmarker.FaceLandmarker:
    """Create a FaceLandmarker with detailed diagnostics and fallbacks."""

    resolved_model_path = model_path.resolve()
    print_debug_info(resolved_model_path)

    try:
        return _build_face_landmarker_from_resolved_path(resolved_model_path)
    except Exception as exc:
        print("\n[ERROR] FaceLandmarker creation failed.")
        print(f"[ERROR] Exception type: {type(exc).__name__}")
        print(f"[ERROR] Exception repr: {exc!r}")
        print("[ERROR] Traceback:")
        print(traceback.format_exc())

        # Windows + Japanese path environments sometimes fail in native code
        # even when the file exists. As a fallback, copy the task file to an
        # ASCII-only temporary path and retry once.
        if any(ord(char) > 127 for char in str(resolved_model_path)):
            temp_dir = Path(tempfile.mkdtemp(prefix="mp_model_"))
            fallback_path = temp_dir / resolved_model_path.name
            shutil.copy2(resolved_model_path, fallback_path)
            print(
                "[DEBUG] Retrying with ASCII-only temporary path: "
                f"{fallback_path}"
            )
            try:
                return _build_face_landmarker_from_resolved_path(fallback_path)
            except Exception as fallback_exc:
                print("\n[ERROR] Fallback FaceLandmarker creation also failed.")
                print(f"[ERROR] Exception type: {type(fallback_exc).__name__}")
                print(f"[ERROR] Exception repr: {fallback_exc!r}")
                print("[ERROR] Traceback:")
                print(traceback.format_exc())
                raise

        raise


def test_face_landmarker(model_path: Path) -> None:
    """Minimal test that loads only the task file."""

    resolved_model_path = model_path.resolve()
    print("\n[TEST] Face Landmarker task load test")

    try:
        with build_face_landmarker(resolved_model_path):
            print("[TEST] Face Landmarker loaded successfully.")
    except Exception as exc:
        print("\n[TEST] Face Landmarker load failed.")
        print(f"[TEST] Exception type: {type(exc).__name__}")
        print(f"[TEST] Exception repr: {exc!r}")
        print("[TEST] Traceback:")
        print(traceback.format_exc())
        raise


def extract_landmarks(
    image_dir: Path,
    model_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Extract landmarks for every image in ``image_dir``.

    Returns
    -------
    landmarks_df:
        Per-landmark table with columns
        ``image_id, landmark_id, x, y, z``.
    failed_df:
        Table containing only failed ``image_id`` values.
    stats:
        Summary counts for console output.
    """

    resolved_model_path = resolve_model_path(model_path)
    image_paths = collect_image_paths(image_dir)

    landmark_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []
    success_count = 0
    total_landmark_count = 0

    if not image_paths:
        stats = {
            "total_images": 0,
            "success_images": 0,
            "failed_images": 0,
            "total_landmarks": 0,
        }
        landmarks_df = pd.DataFrame(columns=LANDMARK_COLUMNS)
        failed_df = pd.DataFrame(columns=FAILED_COLUMNS)
        return landmarks_df, failed_df, stats

    # FaceLandmarker is relatively expensive to construct, so we reuse one
    # instance for the whole batch.
    with build_face_landmarker(resolved_model_path) as landmarker:
        for image_path in tqdm(
            image_paths,
            desc="Extracting landmarks",
            unit="image",
        ):
            image_id = image_id_from_path(image_path, image_dir)

            try:
                image_bgr = load_image_bgr(image_path)
                if image_bgr is None:
                    raise ValueError("Image could not be decoded.")

                # MediaPipe expects RGB input for SRGB images.
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_rgb,
                )

                result = landmarker.detect(mp_image)
                if not result.face_landmarks:
                    failed_rows.append({"image_id": image_id})
                    continue

                # In dental research datasets, one face per image is the common
                # case, so we save the first detected face.
                face_landmarks = result.face_landmarks[0]

                for landmark_id, landmark in enumerate(face_landmarks):
                    landmark_rows.append(
                        {
                            "image_id": image_id,
                            "landmark_id": landmark_id,
                            "x": float(landmark.x),
                            "y": float(landmark.y),
                            "z": float(landmark.z),
                        }
                    )

                success_count += 1
                total_landmark_count += len(face_landmarks)

            except Exception as exc:
                # Keep the batch running even if one file is damaged or malformed.
                failed_rows.append({"image_id": image_id})
                print(f"[WARN] Skipped {image_path.name}: {exc}")

    landmarks_df = pd.DataFrame(landmark_rows, columns=LANDMARK_COLUMNS)
    failed_df = pd.DataFrame(failed_rows, columns=FAILED_COLUMNS)
    stats = {
        "total_images": len(image_paths),
        "success_images": success_count,
        "failed_images": len(failed_rows),
        "total_landmarks": total_landmark_count,
    }
    return landmarks_df, failed_df, stats


def save_csv(
    landmarks_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save landmark and failure tables to CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    landmarks_csv = output_dir / "landmarks.csv"
    failed_csv = output_dir / "failed_images.csv"

    # UTF-8 with BOM is convenient for Excel on Windows.
    landmarks_df.to_csv(landmarks_csv, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")

    return landmarks_csv, failed_csv


def print_summary(stats: dict[str, int], landmarks_csv: Path, failed_csv: Path) -> None:
    """Print a short summary of the extraction results."""

    print("\nExtraction completed.")
    print(f"Total images: {stats['total_images']}")
    print(f"Successful images: {stats['success_images']}")
    print(f"Failed images: {stats['failed_images']}")
    print(f"Total landmarks: {stats['total_landmarks']}")
    print(f"Landmarks CSV: {landmarks_csv.resolve()}")
    print(f"Failed images CSV: {failed_csv.resolve()}")


def main() -> None:
    """Program entry point."""

    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()

    if args.test_face_landmarker:
        test_face_landmarker(model_path)
        return

    landmarks_df, failed_df, stats = extract_landmarks(
        image_dir=image_dir,
        model_path=model_path,
    )
    landmarks_csv, failed_csv = save_csv(landmarks_df, failed_df, output_dir)
    print_summary(stats, landmarks_csv, failed_csv)


if __name__ == "__main__":
    main()
