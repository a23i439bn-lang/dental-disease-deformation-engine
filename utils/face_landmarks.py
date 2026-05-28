from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_FACE_LANDMARKER_PATH = Path("models") / "face_landmarker.task"


@dataclass(frozen=True)
class FaceLandmarkResult:
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray
    detector_mode: str


def load_image_unicode_safe(path: str) -> np.ndarray | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    binary = np.fromfile(str(image_path), dtype=np.uint8)
    if binary.size == 0:
        return None
    return cv2.imdecode(binary, cv2.IMREAD_COLOR)


def save_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {path}")
    encoded.tofile(str(path))


def detect_landmarks_with_tasks_3d(image: np.ndarray, model_path: Path) -> FaceLandmarkResult:
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
    vision_running_mode = mp.tasks.vision.RunningMode

    model_bytes = model_path.read_bytes()
    options = face_landmarker_options(
        base_options=base_options(model_asset_buffer=model_bytes),
        running_mode=vision_running_mode.IMAGE,
        num_faces=1,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with face_landmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected by MediaPipe FaceLandmarker.")

    h, w = image.shape[:2]
    coords_2d: list[list[int]] = []
    coords_3d: list[list[float]] = []
    for lm in result.face_landmarks[0]:
        x_px = float(lm.x * w)
        y_px = float(lm.y * h)
        z_px = float(lm.z * w)
        coords_2d.append([int(round(x_px)), int(round(y_px))])
        coords_3d.append([x_px, y_px, z_px])
    return FaceLandmarkResult(
        landmarks_2d=np.array(coords_2d, dtype=np.int32),
        landmarks_3d=np.array(coords_3d, dtype=np.float32),
        detector_mode="mediapipe_tasks_face_landmarker_3d",
    )


def detect_landmarks_with_tasks(image: np.ndarray, model_path: Path) -> np.ndarray:
    return detect_landmarks_with_tasks_3d(image, model_path).landmarks_2d


def detect_landmarks_3d(image: np.ndarray, face_landmarker_model: str) -> FaceLandmarkResult:
    model_path = Path(face_landmarker_model)
    if model_path.exists():
        try:
            return detect_landmarks_with_tasks_3d(image, model_path)
        except Exception:
            pass

    face_mesh_api = getattr(getattr(mp, "solutions", None), "face_mesh", None)
    if face_mesh_api is None:
        raise RuntimeError("MediaPipe FaceMesh is unavailable and Face Landmarker failed.")

    with face_mesh_api.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No face detected by MediaPipe FaceMesh.")

    h, w = image.shape[:2]
    coords_2d: list[list[int]] = []
    coords_3d: list[list[float]] = []
    for lm in result.multi_face_landmarks[0].landmark:
        x_px = float(lm.x * w)
        y_px = float(lm.y * h)
        z_px = float(lm.z * w)
        coords_2d.append([int(round(x_px)), int(round(y_px))])
        coords_3d.append([x_px, y_px, z_px])
    return FaceLandmarkResult(
        landmarks_2d=np.array(coords_2d, dtype=np.int32),
        landmarks_3d=np.array(coords_3d, dtype=np.float32),
        detector_mode="mediapipe_facemesh_3d",
    )


def detect_landmarks(image: np.ndarray, face_landmarker_model: str) -> tuple[np.ndarray, str]:
    result = detect_landmarks_3d(image, face_landmarker_model)
    return result.landmarks_2d, result.detector_mode.replace("_3d", "")


def parse_severity_values(args: argparse.Namespace) -> list[float]:
    if args.severity_sweep.strip():
        return [float(item.strip()) for item in args.severity_sweep.split(",") if item.strip()]
    return [float(args.severity)]
