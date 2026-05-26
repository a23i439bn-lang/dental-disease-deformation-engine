from __future__ import annotations
# このファイルの役割:
# 疾患ごとの教師変形やルールベース priors をまとめたモジュールです。
# 各疾患でどのランドマークをどう動かすかの基礎ルールを定義します。

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
UPPER_INNER_LIP_INDICES = [82, 13, 312, 311, 310]
LOWER_INNER_LIP_INDICES = [80, 191, 78, 95, 88]
LEFT_FRONT_GAP_INDICES = [82, 81, 80]
RIGHT_FRONT_GAP_INDICES = [312, 311, 310]


STRUCTURAL_DISEASE_SCALES = {
    "maxillary_protrusion": 0.046,
    "diastema": 0.036,
    "open_bite": 0.040,
}

DISEASE_NAME_ALIASES = {
    "maxillary_protrusion": {
        "出っ歯",
        "maxillary_protrusion",
        "protrusion",
        "malocclusion",
    },
    "diastema": {
        "すきっ歯",
        "diastema",
        "spacing",
    },
    "open_bite": {
        "開咬",
        "open_bite",
        "anterior_open_bite",
    },
    "caries": {
        "虫歯",
        "caries",
        "dental_caries",
        "dental caries",
    },
}

''' 疾患名を正規化する関数 '''
def canonicalize_disease_name(name: str) -> str:
    lowered = name.strip().lower()
    for canonical, aliases in DISEASE_NAME_ALIASES.items():
        if name in aliases or lowered in aliases:
            return canonical
    return lowered

''' 疾患名のプリセットから利用可能な疾患名を解決する関数 '''
def resolve_preset_disease_name(name: str, available_names: list[str] | tuple[str, ...] | set[str]) -> str:
    available_lookup = {str(item): str(item) for item in available_names}
    if name in available_lookup:
        return available_lookup[name]

    canonical = canonicalize_disease_name(name)
    for candidate in available_lookup.values():
        if canonicalize_disease_name(candidate) == canonical:
            return candidate
    raise KeyError(f"Unknown disease preset '{name}'")

''' 疾患名のリストから構造的疾患を選択する関数 '''
def select_structural_disease(disease_names: list[str]) -> str | None:
    for name in disease_names:
        canonical = canonicalize_disease_name(name)
        if canonical in STRUCTURAL_DISEASE_SCALES:
            return canonical
    return None

''' 指定された疾患に基づいてテンプレートの変位を構築する関数 '''
def build_template_delta(
    source_landmarks: torch.Tensor,
    disease_name: str,
    severity: torch.Tensor | float,
) -> torch.Tensor:
    if source_landmarks.ndim != 3:
        raise ValueError(f"Expected source_landmarks to have shape (B, 468, 2), got {tuple(source_landmarks.shape)}")

    canonical = canonicalize_disease_name(disease_name)
    delta = torch.zeros_like(source_landmarks)
    mouth = source_landmarks[:, MOUTH_INDICES]
    mouth_width = (mouth[:, :, 0].max(dim=1).values - mouth[:, :, 0].min(dim=1).values).clamp_min(1.0)
    mouth_height = (mouth[:, :, 1].max(dim=1).values - mouth[:, :, 1].min(dim=1).values).clamp_min(1.0)

    if not torch.is_tensor(severity):
        severity_tensor = torch.tensor([severity], device=source_landmarks.device, dtype=source_landmarks.dtype)
    else:
        severity_tensor = severity.to(device=source_landmarks.device, dtype=source_landmarks.dtype)
    if severity_tensor.ndim == 2 and severity_tensor.shape[-1] == 1:
        severity_tensor = severity_tensor.squeeze(-1)
    if severity_tensor.ndim == 0:
        severity_tensor = severity_tensor.unsqueeze(0)
    if severity_tensor.ndim != 1:
        raise ValueError(f"Expected severity to have shape (B,), (B, 1), or scalar, got {tuple(severity_tensor.shape)}")
    if severity_tensor.shape[0] == 1 and source_landmarks.shape[0] > 1:
        severity_tensor = severity_tensor.expand(source_landmarks.shape[0])
    if severity_tensor.shape[0] != source_landmarks.shape[0]:
        raise ValueError("severity batch size must match source_landmarks batch size")

    severity_scale = severity_tensor.clamp(0.0, 1.5).view(-1, 1)

    if canonical == "maxillary_protrusion":
        x_shift = mouth_width.view(-1, 1) * (0.10 * severity_scale + 0.03)
        y_shift = mouth_height.view(-1, 1) * (0.02 * severity_scale)
        delta[:, UPPER_INNER_LIP_INDICES, 0] += x_shift
        delta[:, UPPER_INNER_LIP_INDICES, 1] -= y_shift
        delta[:, [13], 0] += x_shift * 0.35
    elif canonical == "diastema":
        x_shift = mouth_width.view(-1, 1) * (0.07 * severity_scale + 0.02)
        delta[:, LEFT_FRONT_GAP_INDICES, 0] -= x_shift
        delta[:, RIGHT_FRONT_GAP_INDICES, 0] += x_shift
    elif canonical == "open_bite":
        gap = mouth_height.view(-1, 1) * (0.22 * severity_scale + 0.05)
        delta[:, UPPER_INNER_LIP_INDICES, 1] -= gap
        delta[:, LOWER_INNER_LIP_INDICES, 1] += gap
    else:
        raise KeyError(f"Unsupported template disease '{disease_name}' (canonical: '{canonical}')")

    return delta

''' 教師モデルのターゲットランドマークを構築する関数 '''
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
