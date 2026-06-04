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


MENTOLABIAL_FOLD_IDX = [17, 18, 200, 199, 175, 152]
LOWER_LIP_CENTER_IDX = [17, 14, 84, 87, 178, 317, 318, 402]
CHEEK_TRANSITION_IDX = [93, 132, 58, 172, 323, 361, 288, 397]
MIDFACE_STABILIZER_IDX = [2, 5, 94, 97, 98, 168, 195, 197, 326, 327]


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

        face_height = max(metrics["face_height"], 1.0)
        face_width = max(metrics["face_width"], 1.0)
        jaw_span = max(metrics["jaw_span"], 1.0)

        block_down = face_height * (0.010 + 0.022 * severity)
        lower_third_gain = face_height * (0.006 + 0.017 * severity)
        chin_projection = face_width * (0.016 + 0.045 * severity)
        jaw_projection = face_width * (0.010 + 0.030 * severity)
        lip_projection = face_width * (0.008 + 0.025 * severity)
        mentolabial_deepen = face_height * (0.007 + 0.020 * severity)
        cheek_transition = face_width * (0.003 + 0.009 * severity)
        jaw_widen = jaw_span * (0.004 + 0.012 * severity)

        mouth_pivot = np.array(
            [
                metrics["mouth_center_x"],
                metrics["mouth_center_y"] - metrics["mouth_height"] * 0.65,
            ],
            dtype=np.float32,
        )
        sym_axis = float(metrics["mouth_center_x"])
        mandibular_block = self.unique_indices(
            LEFT_LOWER_JAW_IDX,
            RIGHT_LOWER_JAW_IDX,
            CHIN_IDX,
            MIDLINE_LOWER_FACE_IDX,
            LOWER_LIP_IDX,
            MENTOLABIAL_FOLD_IDX,
        )
        self.apply_block_transform(
            src_points,
            dst_points,
            mandibular_block,
            pivot=mouth_pivot,
            translation=(0.0, block_down),
            rotation_deg=1.0 + 2.6 * severity,
            scale=(1.0 + 0.020 * severity, 1.0 + 0.018 * severity),
            weight=0.82,
        )

        for idx in LEFT_LOWER_JAW_IDX:
            dst_points[idx, 0] -= jaw_widen + jaw_projection * 0.38
            dst_points[idx, 1] += block_down * 0.40 + lower_third_gain * 0.32
        for idx in RIGHT_LOWER_JAW_IDX:
            dst_points[idx, 0] += jaw_widen + jaw_projection * 0.38
            dst_points[idx, 1] += block_down * 0.40 + lower_third_gain * 0.32
        for idx in CHIN_IDX:
            direction = -1.0 if landmarks[idx, 0] < sym_axis else 1.0
            centrality = 1.0 - min(abs(float(landmarks[idx, 0]) - sym_axis) / max(metrics["mouth_width"], 1.0), 1.0)
            dst_points[idx, 0] += direction * chin_projection * (0.45 + 0.55 * centrality)
            dst_points[idx, 1] += block_down + lower_third_gain * (0.75 + 0.30 * centrality)
        for idx in MIDLINE_LOWER_FACE_IDX:
            dst_points[idx, 1] += lower_third_gain * 0.62
        for idx in LOWER_LIP_CENTER_IDX:
            direction = -1.0 if landmarks[idx, 0] < sym_axis else 1.0
            dst_points[idx, 0] += direction * lip_projection
            dst_points[idx, 1] += block_down * 0.28
        for idx in MENTOLABIAL_FOLD_IDX:
            dst_points[idx, 1] += mentolabial_deepen
        for idx in [93, 132, 58, 172]:
            dst_points[idx, 0] -= cheek_transition
        for idx in [323, 361, 288, 397]:
            dst_points[idx, 0] += cheek_transition

        anchor_indices = sorted(set(NOSE_ANCHOR_IDX + MIDFACE_STABILIZER_IDX + FACE_OVAL_IDX[:10] + FACE_OVAL_IDX[-10:]))
        control_indices = sorted(
            set(
                anchor_indices
                + LEFT_LOWER_JAW_IDX
                + RIGHT_LOWER_JAW_IDX
                + CHIN_IDX
                + LOWER_LIP_IDX
                + MIDLINE_LOWER_FACE_IDX
                + MENTOLABIAL_FOLD_IDX
                + CHEEK_TRANSITION_IDX
            )
        )
        mask = build_lower_face_mask(image_shape, landmarks, feather)
        weight_map = build_anatomical_weight_map(
            image_shape,
            landmarks,
            [
                (CHIN_IDX, 1.00),
                (LEFT_LOWER_JAW_IDX + RIGHT_LOWER_JAW_IDX, 0.90),
                (MENTOLABIAL_FOLD_IDX, 0.80),
                (LOWER_LIP_IDX, 0.70),
                (CHEEK_TRANSITION_IDX, 0.30),
                (MIDFACE_STABILIZER_IDX, 0.10),
            ],
            base_mask=mask,
            feather=feather,
            base_weight=0.04,
        )
        return TemplateResult(
            canonical_name=self.canonical_name,
            mask=mask,
            src_points=src_points[control_indices],
            dst_points=dst_points[control_indices],
            weight_map=weight_map,
        )
