from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def visualize_delta(
    image: np.ndarray,
    landmarks: np.ndarray,
    delta: np.ndarray,
    save_path: str | Path,
    scale: float = 50.0,
    color: tuple[int, int, int] = (0, 0, 255),
    mouth_indices: list[int] | None = None,
) -> None:
    canvas = np.ascontiguousarray(image.copy())
    selected = range(len(landmarks)) if mouth_indices is None else mouth_indices
    for idx in selected:
        x, y = landmarks[idx]
        dx, dy = delta[idx]
        start = (int(round(x)), int(round(y)))
        end = (int(round(x + dx * scale)), int(round(y + dy * scale)))
        cv2.arrowedLine(canvas, start, end, color, 1, tipLength=0.25)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
