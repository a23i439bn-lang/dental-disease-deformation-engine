from __future__ import annotations

import numpy as np

from disease_templates.base import FaceDiseaseTemplate, TemplateResult
from utils.face_regions import (
    CHIN_IDX,
    FACE_OVAL_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_FACE_OVAL_IDX,
    LOWER_LIP_IDX,
    MIDLINE_LOWER_FACE_IDX,
    NOSE_ANCHOR_IDX,
    RIGHT_LOWER_JAW_IDX,
    build_anatomical_weight_map,
    build_lower_face_mask,
    face_metrics,
)


class MandibularProtrusionTemplate(FaceDiseaseTemplate):
    canonical_name = "mandibular_protrusion"
    aliases = ("underbite", "prognathism", "class_iii")

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

        chin_drop = metrics["face_height"] * (0.014 + 0.034 * severity)
        jaw_widen = metrics["jaw_span"] * (0.010 + 0.030 * severity)
        lower_lip_drop = metrics["face_height"] * (0.005 + 0.010 * severity)
        cheek_widen = metrics["face_width"] * (0.004 + 0.012 * severity)
        chin_forward_fan = metrics["face_width"] * (0.008 + 0.018 * severity)

        mouth_pivot = np.array(
            [
                metrics["mouth_center_x"],
                metrics["mouth_center_y"] - metrics["mouth_height"] * 0.85,
            ],
            dtype=np.float32,
        )
        mandibular_block = self.unique_indices(
            LEFT_LOWER_JAW_IDX,
            RIGHT_LOWER_JAW_IDX,
            CHIN_IDX,
            MIDLINE_LOWER_FACE_IDX,
            LOWER_LIP_IDX,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            mandibular_block,
            pivot=mouth_pivot,
            translation=(0.0, chin_drop * 0.28),
            scale=(1.0 + 0.018 * severity, 1.0 + 0.010 * severity),
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            LEFT_LOWER_JAW_IDX + CHIN_IDX,
            pivot=mouth_pivot,
            rotation_deg=0.9 + 1.2 * severity,
            weight=0.55,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            RIGHT_LOWER_JAW_IDX + CHIN_IDX,
            pivot=mouth_pivot,
            rotation_deg=-(0.9 + 1.2 * severity),
            weight=0.55,
        )

        for idx in LEFT_LOWER_JAW_IDX:
            dst_points[idx, 0] -= jaw_widen
            dst_points[idx, 1] += chin_drop * 0.22
        for idx in RIGHT_LOWER_JAW_IDX:
            dst_points[idx, 0] += jaw_widen
            dst_points[idx, 1] += chin_drop * 0.22
        for idx in CHIN_IDX:
            direction = -1.0 if landmarks[idx, 0] < metrics["mouth_center_x"] else 1.0
            dst_points[idx, 0] += direction * chin_forward_fan
            dst_points[idx, 1] += chin_drop
        for idx in MIDLINE_LOWER_FACE_IDX:
            dst_points[idx, 1] += chin_drop * 0.55
        for idx in LOWER_LIP_IDX:
            direction = -1.0 if landmarks[idx, 0] < metrics["mouth_center_x"] else 1.0
            dst_points[idx, 0] += direction * chin_forward_fan * 0.42
            dst_points[idx, 1] += lower_lip_drop

        for idx in [93, 132, 58]:
            dst_points[idx, 0] -= cheek_widen
        for idx in [323, 361, 288]:
            dst_points[idx, 0] += cheek_widen

        anchor_indices = sorted(set(NOSE_ANCHOR_IDX + FACE_OVAL_IDX[:8] + FACE_OVAL_IDX[-8:]))
        control_indices = sorted(
            set(
                anchor_indices
                + LEFT_LOWER_JAW_IDX
                + RIGHT_LOWER_JAW_IDX
                + CHIN_IDX
                + LOWER_LIP_IDX
                + MIDLINE_LOWER_FACE_IDX
            )
        )
        mask = build_lower_face_mask(image_shape, landmarks, feather)
        weight_map = build_anatomical_weight_map(
            image_shape,
            landmarks,
            [
                (CHIN_IDX + MIDLINE_LOWER_FACE_IDX, 1.00),
                (LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX, 0.72),
                (LOWER_LIP_IDX, 0.62),
                ([93, 132, 58, 323, 361, 288], 0.30),
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
