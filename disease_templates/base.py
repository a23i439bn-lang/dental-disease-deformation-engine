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


class FaceDiseaseTemplate(ABC):
    canonical_name: str
    aliases: tuple[str, ...]

    def matches(self, name: str) -> bool:
        normalized = name.strip().lower().replace(" ", "_")
        return normalized == self.canonical_name or normalized in self.aliases

    @staticmethod
    def clamp_severity(severity: float) -> float:
        return float(np.clip(severity, 0.0, 1.5))

    @abstractmethod
    def build(
        self,
        image_shape: tuple[int, int, int],
        landmarks: np.ndarray,
        severity: float,
        feather: int,
    ) -> TemplateResult:
        raise NotImplementedError
