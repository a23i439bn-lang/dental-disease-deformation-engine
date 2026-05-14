from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MOUTH_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]
ANCHOR_INDICES = [1, 4, 6, 8, 9, 10, 33, 61, 93, 127, 152, 172, 197, 234, 263, 291, 323, 356, 389, 454]
INNER_MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


@dataclass
class DatasetBuilderConfig:
    data_root: str = "data"
    input_dir: str = "data/inputs"
    references_dir: str = "data/references"
    manifest_path: str = "data/research_manifest.json"
    image_size: int = 256
    diseases: tuple[str, ...] = ("虫歯", "出っ歯", "すきっ歯")
    samples_per_image: int = 2


class FaceLandmarkDetector:
    def __init__(self) -> None:
        self.face_mesh = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def __call__(self, image: Image.Image) -> np.ndarray:
        if self.face_mesh is not None:
            result = self.face_mesh.process(np.array(image))
            if result.multi_face_landmarks:
                width, height = image.size
                landmarks = result.multi_face_landmarks[0].landmark
                return np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)
        return heuristic_landmarks(image)

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()


def heuristic_landmarks(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces: list[Any] | tuple[Any, ...] = []
    if os.environ.get("ENABLE_HAAR_FALLBACK", "0") == "1":
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    width, height = image.size
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    else:
        x = int(width * 0.18)
        y = int(height * 0.12)
        w = int(width * 0.64)
        h = int(height * 0.76)

    landmarks = np.zeros((468, 2), dtype=np.float32)
    for idx in ANCHOR_INDICES:
        landmarks[idx] = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    mouth_center = np.array([x + w * 0.5, y + h * 0.62], dtype=np.float32)
    mouth_radius = np.array([w * 0.15, h * 0.045], dtype=np.float32)
    for offset, idx in enumerate(MOUTH_INDICES):
        angle = (2.0 * np.pi * offset) / len(MOUTH_INDICES)
        landmarks[idx] = mouth_center + np.array([np.cos(angle) * mouth_radius[0], np.sin(angle) * mouth_radius[1]], dtype=np.float32)
    inner_radius = mouth_radius * np.array([0.72, 0.52], dtype=np.float32)
    for offset, idx in enumerate(INNER_MOUTH_INDICES):
        angle = (2.0 * np.pi * offset) / len(INNER_MOUTH_INDICES)
        landmarks[idx] = mouth_center + np.array([np.cos(angle) * inner_radius[0], np.sin(angle) * inner_radius[1]], dtype=np.float32)
    return landmarks


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return resized


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    center = landmarks.mean(axis=0, keepdims=True)
    scale = np.maximum(landmarks.max(axis=0, keepdims=True) - landmarks.min(axis=0, keepdims=True), 1.0)
    return ((landmarks - center) / scale).astype(np.float32)


def build_multi_hot(disease_names: list[str], known_diseases: list[str]) -> np.ndarray:
    multi_hot = np.zeros(len(known_diseases), dtype=np.float32)
    for disease in disease_names:
        if disease in known_diseases:
            multi_hot[known_diseases.index(disease)] = 1.0
    return multi_hot


def synthesize_target_image(image: Image.Image, disease_names: list[str], severity: float) -> Image.Image:
    rgb = np.array(image).astype(np.float32)
    output = rgb.copy()
    if "虫歯" in disease_names:
        output[:, :, 0] *= 1.0 - 0.12 * severity
        output[:, :, 1] *= 1.0 - 0.06 * severity
        output[:, :, 2] *= 1.0 - 0.02 * severity
    if "出っ歯" in disease_names:
        output = np.roll(output, shift=max(1, int(3 * severity)), axis=1)
    if "すきっ歯" in disease_names:
        center = output.shape[1] // 2
        gap = max(1, int(4 * severity))
        output[:, center - gap:center + gap] = 255.0
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def build_manifest(config: DatasetBuilderConfig) -> dict[str, Any]:
    input_dir = Path(config.input_dir)
    references_dir = Path(config.references_dir)
    records: list[dict[str, Any]] = []
    image_paths = sorted([path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    rng = random.Random(42)
    for image_path in image_paths:
        for _ in range(config.samples_per_image):
            disease_count = rng.randint(1, min(2, len(config.diseases)))
            diseases = rng.sample(list(config.diseases), k=disease_count)
            severity = round(rng.uniform(0.2, 1.0), 3)
            record = {
                "image_path": str(image_path.resolve()),
                "diseases": diseases,
                "severity": severity,
                "prompt": ", ".join(diseases),
                "reference_paths": [
                    str(path.resolve())
                    for disease in diseases
                    for path in (references_dir / disease).iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ],
            }
            records.append(record)
    manifest = {"records": records, "diseases": list(config.diseases)}
    Path(config.manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class DiseaseResearchDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int = 256) -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.records = manifest["records"]
        self.diseases = manifest["diseases"]
        self.image_size = image_size
        self.detector = FaceLandmarkDetector()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str] | str]:
        record = self.records[index]
        image = resize_image(Image.open(record["image_path"]).convert("RGB"), self.image_size)
        target_image = resize_image(synthesize_target_image(image, record["diseases"], record["severity"]), self.image_size)
        landmarks = self.detector(image)
        sample = {
            "image": torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1),
            "target_image": torch.from_numpy(np.array(target_image).astype(np.float32) / 255.0).permute(2, 0, 1),
            "landmarks": torch.from_numpy(normalize_landmarks(landmarks)),
            "landmarks_px": torch.from_numpy(landmarks.astype(np.float32)),
            "multi_hot": torch.from_numpy(build_multi_hot(record["diseases"], self.diseases)),
            "severity": torch.tensor([record["severity"]], dtype=torch.float32),
            "disease_names": record["diseases"],
            "prompt": record["prompt"],
            "image_path": record["image_path"],
        }
        return sample

    def close(self) -> None:
        self.detector.close()


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["image", "target_image", "landmarks", "landmarks_px", "multi_hot", "severity"]
    collated: dict[str, Any] = {}
    for key in tensor_keys:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    collated["disease_names"] = [sample["disease_names"] for sample in batch]
    collated["prompt"] = [sample["prompt"] for sample in batch]
    collated["image_path"] = [sample["image_path"] for sample in batch]
    return collated
