'''This module provides functions to visualize the delta of facial landmarks on an image. It includes two main functions: `visualize_delta` for drawing arrows representing the delta, and `visualize_landmark_overlay` for overlaying the original and target landmarks on the image. Both functions save the resulting visualization to a specified path.'''
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

'''画像上の顔ランドマークの差分（デルタ）を
可視化する関数を提供するモジュールです。主に2つの関数があります'''
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

'''ランドマークのオーバーレイを可視化する関数です。
元のランドマークとデルタを適用したターゲットランドマークを
画像上に描画し、指定されたパスに保存します。'''
def visualize_landmark_overlay(
    image: np.ndarray,
    landmarks: np.ndarray,
    delta: np.ndarray,
    save_path: str | Path,
    mouth_indices: list[int] | None = None,
) -> None:
    canvas = np.ascontiguousarray(image.copy())
    selected = range(len(landmarks)) if mouth_indices is None else mouth_indices
    target = landmarks + delta
    for idx in selected:
        x, y = landmarks[idx]
        tx, ty = target[idx]
        start = (int(round(x)), int(round(y)))
        end = (int(round(tx)), int(round(ty)))
        cv2.circle(canvas, start, 2, (0, 255, 0), -1)
        cv2.circle(canvas, end, 2, (255, 64, 64), -1)
        cv2.arrowedLine(canvas, start, end, (255, 0, 0), 1, tipLength=0.25)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
