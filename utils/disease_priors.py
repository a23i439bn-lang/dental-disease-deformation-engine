from __future__ import annotations

import math

import torch


MOUTH_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]
ANCHOR_INDICES = [1, 4, 6, 8, 9, 10, 33, 61, 93, 127, 152, 172, 197, 234, 263, 291, 323, 356, 389, 454]
INNER_MOUTH_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


STRUCTURAL_DISEASE_SCALES = {
    "maxillary_protrusion": 0.046,
    "diastema": 0.036,
}

DISEASE_NAME_ALIASES = {
    "maxillary_protrusion": {
        "出っ歯",
        "蜃ｺ縺｣豁ｯ",
        "maxillary_protrusion",
        "protrusion",
        "open_bite",
        "malocclusion",
    },
    "diastema": {
        "すきっ歯",
        "縺吶″縺｣豁ｯ",
        "diastema",
        "spacing",
    },
    "caries": {
        "虫歯",
        "陌ｫ豁ｯ",
        "caries",
        "dental_caries",
    },
}


def canonicalize_disease_name(name: str) -> str:
    lowered = name.strip().lower()
    for canonical, aliases in DISEASE_NAME_ALIASES.items():
        if name in aliases or lowered in aliases:
            return canonical
    return lowered


def select_structural_disease(disease_names: list[str]) -> str | None:
    for name in disease_names:
        canonical = canonicalize_disease_name(name)
        if canonical in STRUCTURAL_DISEASE_SCALES:
            return canonical
    return None


def build_teacher_target_landmarks(
    source_landmarks: torch.Tensor,
    disease_names_batch: list[list[str]],
    severity: torch.Tensor,
) -> torch.Tensor:
    if source_landmarks.ndim != 3:
        raise ValueError(f"Expected source_landmarks to have shape (B, 468, 2), got {tuple(source_landmarks.shape)}")
    if severity.ndim == 2 and severity.shape[-1] == 1:
        severity = severity.squeeze(-1)
    if severity.ndim != 1:
        raise ValueError(f"Expected severity to have shape (B,) or (B, 1), got {tuple(severity.shape)}")

    target_landmarks = source_landmarks.clone()
    mouth_idx = torch.tensor(MOUTH_INDICES, device=source_landmarks.device, dtype=torch.long)
    x_offsets = torch.sin(
        torch.linspace(0.0, math.pi * 2.0, steps=len(MOUTH_INDICES), device=source_landmarks.device, dtype=source_landmarks.dtype)
    ).view(1, len(MOUTH_INDICES), 1)
    zeros = torch.zeros_like(x_offsets)
    offset_basis = torch.cat([x_offsets, zeros], dim=-1)

    for batch_idx, disease_names in enumerate(disease_names_batch):
        structural = select_structural_disease(disease_names)
        if structural is None:
            continue
        scale_xy = STRUCTURAL_DISEASE_SCALES[structural]
        source_mouth = source_landmarks[batch_idx, mouth_idx]
        mouth_scale = (source_mouth.max(dim=0).values - source_mouth.min(dim=0).values).clamp_min(1.0)
        target_landmarks[batch_idx, mouth_idx] = source_mouth + offset_basis[0] * mouth_scale.view(1, 2) * scale_xy * severity[batch_idx]
    return target_landmarks
