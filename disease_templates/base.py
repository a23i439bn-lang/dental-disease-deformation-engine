from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TemplateResult:
    canonical_name: str
    mask: np.ndarray
    src_points: np.ndarray
    dst_points: np.ndarray
    weight_map: np.ndarray | None = None


class FaceDiseaseTemplate(ABC):
    canonical_name: str
    aliases: tuple[str, ...]

    def matches(self, name: str) -> bool:
        normalized = name.strip().lower().replace(" ", "_")
        return normalized == self.canonical_name or normalized in self.aliases

    @staticmethod
    def clamp_severity(severity: float) -> float:
        return float(np.clip(severity, 0.0, 1.5))

    @staticmethod
    def unique_indices(*groups: list[int]) -> list[int]:
        return sorted(set(idx for group in groups for idx in group))

    @staticmethod
    def centroid(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros(2, dtype=np.float32)
        return points.astype(np.float32).mean(axis=0)

    @staticmethod
    def apply_block_transform(
        source_landmarks: np.ndarray,
        target_landmarks: np.ndarray,
        indices: list[int],
        pivot: np.ndarray,
        translation: np.ndarray | tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        scale: np.ndarray | tuple[float, float] = (1.0, 1.0),
        weight: float = 1.0,
    ) -> None:
        if not indices or weight <= 0.0:
            return

        indices = sorted(set(indices))
        pts = source_landmarks[indices].astype(np.float32)
        pivot = np.asarray(pivot, dtype=np.float32)
        translation = np.asarray(translation, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)

        angle = np.deg2rad(rotation_deg)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        local = (pts - pivot) * scale
        transformed = local @ rot.T + pivot + translation
        target_landmarks[indices] += (transformed - pts) * float(weight)

    @abstractmethod
    def build(
        self,
        image_shape: tuple[int, int, int],
        landmarks: np.ndarray,
        severity: float,
        feather: int,
    ) -> TemplateResult:
        raise NotImplementedError
