from __future__ import annotations

import numpy as np

from disease_templates.base import FaceDiseaseTemplate, TemplateResult
from utils.face_regions import (
    CHIN_IDX,
    FACE_OVAL_IDX,
    LEFT_CHIN_SIDE_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MIDLINE_LOWER_FACE_IDX,
    NOSE_ANCHOR_IDX,
    RIGHT_CHIN_SIDE_IDX,
    RIGHT_LOWER_JAW_IDX,
    build_anatomical_weight_map,
    build_lower_face_mask,
    face_metrics,
)


class ChinDeviationRightTemplate(FaceDiseaseTemplate):
    canonical_name = "chin_deviation_right"
    aliases = ("menton_deviation_right", "chin_asymmetry_right")

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

        chin_shift = metrics["face_width"] * (0.014 + 0.038 * severity)
        jaw_shift = metrics["face_width"] * (0.010 + 0.028 * severity)
        cant_drop = metrics["face_height"] * (0.004 + 0.010 * severity)

        lower_face_pivot = np.array(
            [
                metrics["mouth_center_x"],
                metrics["mouth_center_y"] - metrics["mouth_height"] * 0.75,
            ],
            dtype=np.float32,
        )
        lower_block = self.unique_indices(
            LEFT_LOWER_JAW_IDX,
            RIGHT_LOWER_JAW_IDX,
            LEFT_CHIN_SIDE_IDX,
            RIGHT_CHIN_SIDE_IDX,
            CHIN_IDX,
            MIDLINE_LOWER_FACE_IDX,
            LOWER_LIP_IDX,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            lower_block,
            pivot=lower_face_pivot,
            translation=(chin_shift * 0.20, cant_drop * 0.22),
            rotation_deg=-(1.2 + 2.4 * severity),
            weight=0.78,
        )

        for idx in CHIN_IDX + MIDLINE_LOWER_FACE_IDX:
            dst_points[idx, 0] += chin_shift
        for idx in RIGHT_LOWER_JAW_IDX + RIGHT_CHIN_SIDE_IDX:
            dst_points[idx, 0] += jaw_shift
            dst_points[idx, 1] -= cant_drop * 0.35
        for idx in LEFT_LOWER_JAW_IDX + LEFT_CHIN_SIDE_IDX:
            dst_points[idx, 0] += jaw_shift * 0.35
            dst_points[idx, 1] += cant_drop
        for idx in LOWER_LIP_IDX:
            dst_points[idx, 0] += chin_shift * 0.40

        anchor_indices = sorted(set(NOSE_ANCHOR_IDX + FACE_OVAL_IDX[:8] + FACE_OVAL_IDX[-8:]))
        control_indices = sorted(
            set(
                anchor_indices
                + LEFT_LOWER_JAW_IDX
                + RIGHT_LOWER_JAW_IDX
                + LEFT_CHIN_SIDE_IDX
                + RIGHT_CHIN_SIDE_IDX
                + CHIN_IDX
                + MIDLINE_LOWER_FACE_IDX
                + LOWER_LIP_IDX
            )
        )
        mask = build_lower_face_mask(image_shape, landmarks, feather)
        weight_map = build_anatomical_weight_map(
            image_shape,
            landmarks,
            [
                (CHIN_IDX + MIDLINE_LOWER_FACE_IDX, 1.00),
                (RIGHT_CHIN_SIDE_IDX + RIGHT_LOWER_JAW_IDX, 0.78),
                (LEFT_CHIN_SIDE_IDX + LEFT_LOWER_JAW_IDX, 0.52),
                (LOWER_LIP_IDX, 0.58),
            ],
            base_mask=mask,
            feather=feather,
            base_weight=0.10,
        )
        return TemplateResult(
            canonical_name=self.canonical_name,
            mask=mask,
            src_points=src_points[control_indices],
            dst_points=dst_points[control_indices],
            weight_map=weight_map,
        )
