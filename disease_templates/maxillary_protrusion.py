from __future__ import annotations

import numpy as np

from disease_templates.base import FaceDiseaseTemplate, TemplateResult
from utils.face_regions import (
    CHIN_IDX,
    FACE_OVAL_IDX,
    LEFT_LOWER_JAW_IDX,
    LOWER_LIP_IDX,
    MOUTH_OUTER_IDX,
    NOSE_ANCHOR_IDX,
    NOSE_BASE_IDX,
    RIGHT_LOWER_JAW_IDX,
    UPPER_LIP_IDX,
    UPPER_MOUTH_IDX,
    build_anatomical_weight_map,
    build_convex_mask,
    face_metrics,
)


PHILTRUM_IDX = [0, 37, 39, 40, 185, 267, 269, 270, 409]
SUBNASALE_IDX = [2, 5, 94, 97, 98, 168, 326, 327]
MOUTH_PROMINENCE_IDX = [61, 291, 13, 14, 17, 78, 95, 88, 308, 324, 318]
NASAL_BRIDGE_ANCHOR_IDX = [1, 4, 6, 19, 195, 197]
MANDIBULAR_BORDER_IDX = [172, 136, 150, 149, 176, 148, 377, 400, 378, 379, 365, 397]


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

        face_width = max(metrics["face_width"], 1.0)
        face_height = max(metrics["face_height"], 1.0)
        mouth_width = max(metrics["mouth_width"], 1.0)

        upper_lip_forward = face_width * (0.014 + 0.042 * severity)
        philtrum_forward = face_width * (0.009 + 0.026 * severity)
        subnasale_forward = face_width * (0.004 + 0.012 * severity)
        mouth_block_forward = face_width * (0.010 + 0.028 * severity)
        mouth_block_drop = face_height * (0.003 + 0.008 * severity)
        chin_retrusion = face_width * (0.010 + 0.030 * severity)
        mandibular_retrusion = face_width * (0.004 + 0.014 * severity)
        mandibular_deficiency_lift = face_height * (0.002 + 0.006 * severity)

        subnasale_pivot = self.centroid(landmarks[NOSE_BASE_IDX])
        sym_axis = float(metrics["mouth_center_x"])
        maxillary_block = self.unique_indices(
            UPPER_LIP_IDX,
            UPPER_MOUTH_IDX,
            PHILTRUM_IDX,
            SUBNASALE_IDX,
            MOUTH_OUTER_IDX,
            MOUTH_PROMINENCE_IDX,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            maxillary_block,
            pivot=subnasale_pivot,
            translation=(0.0, mouth_block_drop),
            scale=(1.0 + 0.022 * severity, 1.0 + 0.010 * severity),
            weight=0.82,
        )
        left_maxillary = [idx for idx in maxillary_block if landmarks[idx, 0] < metrics["mouth_center_x"]]
        right_maxillary = [idx for idx in maxillary_block if landmarks[idx, 0] >= metrics["mouth_center_x"]]
        self.apply_block_transform(
            src_points,
            dst_points,
            left_maxillary,
            pivot=subnasale_pivot,
            rotation_deg=0.55 + 1.05 * severity,
            weight=0.38,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            right_maxillary,
            pivot=subnasale_pivot,
            rotation_deg=-(0.55 + 1.05 * severity),
            weight=0.38,
        )

        for idx in UPPER_LIP_IDX:
            direction = 1.0 if landmarks[idx, 0] >= sym_axis else -1.0
            centrality = 1.0 - min(abs(float(landmarks[idx, 0]) - sym_axis) / mouth_width, 1.0)
            dst_points[idx, 0] += direction * upper_lip_forward * (0.48 + 0.52 * centrality)
            dst_points[idx, 1] += mouth_block_drop * (0.65 + 0.25 * centrality)
        for idx in PHILTRUM_IDX:
            direction = 1.0 if landmarks[idx, 0] >= sym_axis else -1.0
            centrality = 1.0 - min(abs(float(landmarks[idx, 0]) - sym_axis) / mouth_width, 1.0)
            dst_points[idx, 0] += direction * philtrum_forward * (0.45 + 0.55 * centrality)
            dst_points[idx, 1] += mouth_block_drop * 0.45
        for idx in SUBNASALE_IDX:
            direction = 1.0 if landmarks[idx, 0] >= sym_axis else -1.0
            centrality = 1.0 - min(abs(float(landmarks[idx, 0]) - sym_axis) / mouth_width, 1.0)
            dst_points[idx, 0] += direction * subnasale_forward * (0.35 + 0.65 * centrality)
            dst_points[idx, 1] += mouth_block_drop * 0.18
        for idx in self.unique_indices(MOUTH_OUTER_IDX, MOUTH_PROMINENCE_IDX):
            direction = 1.0 if landmarks[idx, 0] >= sym_axis else -1.0
            dst_points[idx, 0] += direction * mouth_block_forward
            dst_points[idx, 1] += mouth_block_drop * 0.55
        for idx in CHIN_IDX:
            direction = -1.0 if landmarks[idx, 0] >= sym_axis else 1.0
            dst_points[idx, 0] += direction * chin_retrusion
            dst_points[idx, 1] -= mandibular_deficiency_lift
        for idx in self.unique_indices(LEFT_LOWER_JAW_IDX, RIGHT_LOWER_JAW_IDX, MANDIBULAR_BORDER_IDX):
            direction = 1.0 if landmarks[idx, 0] < sym_axis else -1.0
            dst_points[idx, 0] += direction * mandibular_retrusion
            dst_points[idx, 1] -= mandibular_deficiency_lift * 0.55

        anchor_indices = sorted(set(FACE_OVAL_IDX[:10] + FACE_OVAL_IDX[-10:] + NOSE_ANCHOR_IDX + NASAL_BRIDGE_ANCHOR_IDX))
        control_indices = sorted(
            set(
                anchor_indices
                + UPPER_LIP_IDX
                + UPPER_MOUTH_IDX
                + PHILTRUM_IDX
                + SUBNASALE_IDX
                + MOUTH_OUTER_IDX
                + MOUTH_PROMINENCE_IDX
                + LOWER_LIP_IDX
                + CHIN_IDX
                + LEFT_LOWER_JAW_IDX
                + RIGHT_LOWER_JAW_IDX
                + MANDIBULAR_BORDER_IDX
            )
        )
        mask = build_convex_mask(
            image_shape,
            [
                landmarks[SUBNASALE_IDX],
                landmarks[UPPER_LIP_IDX],
                landmarks[MOUTH_OUTER_IDX],
                landmarks[CHIN_IDX],
                landmarks[LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX],
            ],
            feather,
        )
        weight_map = build_anatomical_weight_map(
            image_shape,
            landmarks,
            [
                (UPPER_LIP_IDX + UPPER_MOUTH_IDX, 1.00),
                (PHILTRUM_IDX, 0.86),
                (MOUTH_OUTER_IDX + MOUTH_PROMINENCE_IDX, 0.78),
                (SUBNASALE_IDX, 0.50),
                (CHIN_IDX, 0.44),
                (LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX + MANDIBULAR_BORDER_IDX, 0.30),
                (NASAL_BRIDGE_ANCHOR_IDX, 0.05),
            ],
            base_mask=mask,
            feather=feather,
            base_weight=0.05,
        )
        return TemplateResult(
            canonical_name=self.canonical_name,
            mask=mask,
            src_points=src_points[control_indices],
            dst_points=dst_points[control_indices],
            weight_map=weight_map,
        )
