from __future__ import annotations

import numpy as np

from disease_templates.base import FaceDiseaseTemplate, TemplateResult
from utils.face_regions import (
    CHIN_IDX,
    FACE_OVAL_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MOUTH_CORNER_IDX,
    MOUTH_OUTER_IDX,
    NOSE_ANCHOR_IDX,
    RIGHT_LOWER_JAW_IDX,
    build_occlusal_cant_mask,
    face_metrics,
)


class OcclusalPlaneCantTemplate(FaceDiseaseTemplate):
    canonical_name = "occlusal_plane_cant"
    aliases = ("occlusal_cant", "cant")

    def __init__(self, direction: float = -1.0, canonical_name: str | None = None, aliases: tuple[str, ...] = ()) -> None:
        self.direction = direction
        if canonical_name is not None:
            self.canonical_name = canonical_name
        if aliases:
            self.aliases = aliases

    def build(
        self,
        image_shape: tuple[int, int, int],
        landmarks: np.ndarray,
        severity: float,
        feather: int,
    ) -> TemplateResult:
        severity = self.clamp_severity(severity)
        src_points = landmarks.astype(np.float32).copy()
        dst_points = src_points.copy()
        metrics = face_metrics(landmarks)

        cant_amplitude = metrics["mouth_height"] * (0.18 + 0.22 * severity)
        chin_compensation = metrics["face_height"] * (0.006 + 0.012 * severity)
        jaw_compensation = metrics["face_height"] * (0.004 + 0.008 * severity)
        mouth_center_x = metrics["mouth_center_x"]

        for idx in MOUTH_OUTER_IDX + LOWER_LIP_IDX:
            x_offset = (landmarks[idx, 0] - mouth_center_x) / max(metrics["mouth_width"], 1.0)
            dst_points[idx, 1] += self.direction * x_offset * cant_amplitude

        left_sign = 1.0 if self.direction > 0.0 else -1.0
        right_sign = -left_sign
        for idx in LEFT_LOWER_JAW_IDX:
            dst_points[idx, 1] += left_sign * jaw_compensation
        for idx in RIGHT_LOWER_JAW_IDX:
            dst_points[idx, 1] += right_sign * jaw_compensation
        for idx in CHIN_IDX:
            dst_points[idx, 1] += right_sign * chin_compensation * 0.5
        dst_points[MOUTH_CORNER_IDX[0], 1] += left_sign * cant_amplitude * 0.28
        dst_points[MOUTH_CORNER_IDX[1], 1] += right_sign * cant_amplitude * 0.28

        anchor_indices = sorted(set(NOSE_ANCHOR_IDX + FACE_OVAL_IDX[:10] + FACE_OVAL_IDX[-10:]))
        control_indices = sorted(
            set(anchor_indices + MOUTH_OUTER_IDX + LOWER_LIP_IDX + MOUTH_CORNER_IDX + CHIN_IDX + LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX)
        )
        return TemplateResult(
            canonical_name=self.canonical_name,
            mask=build_occlusal_cant_mask(image_shape, landmarks, feather),
            src_points=src_points[control_indices],
            dst_points=dst_points[control_indices],
        )
