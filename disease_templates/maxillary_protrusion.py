from __future__ import annotations

import numpy as np

from disease_templates.base import FaceDiseaseTemplate, TemplateResult
from utils.face_regions import (
    CHIN_IDX,
    FACE_OVAL_IDX,
    NOSE_ANCHOR_IDX,
    NOSE_BASE_IDX,
    UPPER_LIP_IDX,
    UPPER_MOUTH_IDX,
    build_anatomical_weight_map,
    build_midface_mask,
    face_metrics,
)


class MaxillaryProtrusionTemplate(FaceDiseaseTemplate):
    canonical_name = "maxillary_protrusion"
    aliases = ("overjet", "buck_teeth")

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

        forward_bulge = metrics["face_width"] * (0.010 + 0.028 * severity)
        upper_lip_drop = metrics["face_height"] * (0.004 + 0.008 * severity)
        philtrum_forward = metrics["face_width"] * (0.004 + 0.010 * severity)
        chin_compensate = metrics["face_width"] * (0.003 + 0.010 * severity)

        subnasale_pivot = self.centroid(landmarks[NOSE_BASE_IDX])
        maxillary_block = self.unique_indices(UPPER_LIP_IDX, UPPER_MOUTH_IDX, NOSE_BASE_IDX)
        self.apply_block_transform(
            src_points,
            dst_points,
            maxillary_block,
            pivot=subnasale_pivot,
            translation=(0.0, upper_lip_drop * 0.22),
            scale=(1.0 + 0.012 * severity, 1.0 + 0.006 * severity),
        )
        left_maxillary = [idx for idx in maxillary_block if landmarks[idx, 0] < metrics["mouth_center_x"]]
        right_maxillary = [idx for idx in maxillary_block if landmarks[idx, 0] >= metrics["mouth_center_x"]]
        self.apply_block_transform(
            src_points,
            dst_points,
            left_maxillary,
            pivot=subnasale_pivot,
            rotation_deg=0.45 + 0.65 * severity,
            weight=0.45,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            right_maxillary,
            pivot=subnasale_pivot,
            rotation_deg=-(0.45 + 0.65 * severity),
            weight=0.45,
        )

        for idx in UPPER_LIP_IDX:
            direction = 1.0 if landmarks[idx, 0] >= metrics["mouth_center_x"] else -1.0
            dst_points[idx, 0] += direction * forward_bulge
            dst_points[idx, 1] += upper_lip_drop
        for idx in UPPER_MOUTH_IDX:
            direction = 1.0 if landmarks[idx, 0] >= metrics["mouth_center_x"] else -1.0
            dst_points[idx, 0] += direction * forward_bulge * 0.75
        for idx in NOSE_BASE_IDX:
            direction = 1.0 if landmarks[idx, 0] >= metrics["mouth_center_x"] else -1.0
            dst_points[idx, 0] += direction * philtrum_forward
            dst_points[idx, 1] += upper_lip_drop * 0.35
        for idx in CHIN_IDX:
            direction = -1.0 if landmarks[idx, 0] >= metrics["mouth_center_x"] else 1.0
            dst_points[idx, 0] += direction * chin_compensate

        anchor_indices = sorted(set(FACE_OVAL_IDX[:10] + FACE_OVAL_IDX[-10:] + NOSE_ANCHOR_IDX))
        control_indices = sorted(
            set(anchor_indices + UPPER_LIP_IDX + UPPER_MOUTH_IDX + NOSE_BASE_IDX + CHIN_IDX + [61, 291, 13, 0, 17])
        )
        mask = build_midface_mask(image_shape, landmarks, feather)
        weight_map = build_anatomical_weight_map(
            image_shape,
            landmarks,
            [
                (UPPER_LIP_IDX + UPPER_MOUTH_IDX, 1.00),
                (NOSE_BASE_IDX, 0.58),
                ([61, 291, 13, 0, 17], 0.42),
                (CHIN_IDX, 0.20),
            ],
            base_mask=mask,
            feather=feather,
            base_weight=0.08,
        )
        return TemplateResult(
            canonical_name=self.canonical_name,
            mask=mask,
            src_points=src_points[control_indices],
            dst_points=dst_points[control_indices],
            weight_map=weight_map,
        )
