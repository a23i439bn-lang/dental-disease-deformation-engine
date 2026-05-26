from __future__ import annotations

import cv2
import numpy as np


FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LOWER_FACE_OVAL_IDX = [
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234,
]
CHIN_IDX = [152, 377, 400, 378, 379]
MOUTH_OUTER_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
LOWER_LIP_IDX = [17, 84, 181, 91, 146, 61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
UPPER_LIP_IDX = [0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 185]
UPPER_MOUTH_IDX = [0, 267, 269, 270, 409, 291, 61, 185, 40, 39, 37]
NOSE_BASE_IDX = [2, 5, 94, 97, 98, 327, 326, 168]
NOSE_ANCHOR_IDX = [1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 326, 327]
LEFT_LOWER_JAW_IDX = [234, 93, 132, 58, 172, 136, 150, 149, 176]
RIGHT_LOWER_JAW_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400]
MIDLINE_LOWER_FACE_IDX = [152, 377, 17]
LEFT_CHIN_SIDE_IDX = [149, 150, 136, 172]
RIGHT_CHIN_SIDE_IDX = [378, 379, 365, 397]
MOUTH_CORNER_IDX = [61, 291]


def _blur_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return mask
    kernel = feather if feather % 2 == 1 else feather + 1
    return cv2.GaussianBlur(mask, (kernel, kernel), 0)


def build_convex_mask(
    image_shape: tuple[int, int, int],
    landmark_groups: list[np.ndarray],
    feather: int,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    region_points = np.concatenate(landmark_groups, axis=0).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(region_points), 255)
    return _blur_mask(mask, feather)


def build_lower_face_mask(image_shape: tuple[int, int, int], landmarks: np.ndarray, feather: int) -> np.ndarray:
    return build_convex_mask(
        image_shape,
        [
            landmarks[LOWER_FACE_OVAL_IDX],
            landmarks[MOUTH_OUTER_IDX][::-1],
        ],
        feather,
    )


def build_midface_mask(image_shape: tuple[int, int, int], landmarks: np.ndarray, feather: int) -> np.ndarray:
    return build_convex_mask(
        image_shape,
        [
            landmarks[NOSE_BASE_IDX],
            landmarks[UPPER_LIP_IDX],
            landmarks[[234, 93, 132, 58, 172, 397, 365, 379, 323, 454]],
        ],
        feather,
    )


def build_occlusal_cant_mask(image_shape: tuple[int, int, int], landmarks: np.ndarray, feather: int) -> np.ndarray:
    return build_convex_mask(
        image_shape,
        [
            landmarks[MOUTH_OUTER_IDX],
            landmarks[LOWER_FACE_OVAL_IDX],
            landmarks[NOSE_BASE_IDX],
        ],
        feather,
    )


def face_metrics(landmarks: np.ndarray) -> dict[str, float]:
    face_oval = landmarks[FACE_OVAL_IDX]
    lower_face = landmarks[LOWER_FACE_OVAL_IDX]
    mouth_outer = landmarks[MOUTH_OUTER_IDX]
    return {
        "face_width": float(face_oval[:, 0].max() - face_oval[:, 0].min()),
        "face_height": float(face_oval[:, 1].max() - face_oval[:, 1].min()),
        "jaw_span": float(lower_face[:, 0].max() - lower_face[:, 0].min()),
        "mouth_width": float(mouth_outer[:, 0].max() - mouth_outer[:, 0].min()),
        "mouth_height": float(mouth_outer[:, 1].max() - mouth_outer[:, 1].min()),
        "mouth_center_x": float(np.mean(mouth_outer[:, 0])),
        "mouth_center_y": float(np.mean(mouth_outer[:, 1])),
    }
