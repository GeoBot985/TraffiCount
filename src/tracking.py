from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from obj_detection import Detection

MERGE_LABELS = {"truck", "bus", "train"}


def box_to_cxcywh(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return cx, cy, w, h


def cxcywh_to_box(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def expand_box(
    box: Tuple[float, float, float, float],
    ratio: float,
    frame_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box_to_cxcywh(box)
    w *= 1.0 + ratio
    h *= 1.0 + ratio
    x1, y1, x2, y2 = cxcywh_to_box(cx, cy, w, h)
    height, width = frame_shape
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width * 1.0, x2))
    y2 = max(0.0, min(height * 1.0, y2))
    return (x1, y1, x2, y2)


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def should_merge_boxes(
    label: str,
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> bool:
    if compute_iou(box_a, box_b) >= 0.2:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    center_dist = hypot(((ax1 + ax2) / 2) - ((bx1 + bx2) / 2), ((ay1 + ay2) / 2) - ((by1 + by2) / 2))
    if center_dist > 180.0:
        return False

    overlap_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    width_min = min(ax2 - ax1, bx2 - bx1)
    height_min = min(ay2 - ay1, by2 - by1)

    if overlap_x >= 0.2 * width_min or overlap_y >= 0.2 * height_min:
        return True

    if label in {"truck", "train"}:
        # Long articulated vehicles often produce adjacent boxes (tractor + trailer). Treat them
        # as a single object when there is strong vertical overlap and only a modest horizontal gap.
        vertical_overlap_ratio = overlap_y / max(height_min, 1.0)
        gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
        avg_width = (ax2 - ax1 + bx2 - bx1) / 2.0
        if vertical_overlap_ratio >= 0.4 and gap_x <= max(100.0, avg_width * 0.35):
            return True

    return center_dist <= 120.0


def merge_vehicle_detections(detections: List[Detection]) -> List[Detection]:
    if not detections:
        return []

    merged: List[Detection] = []
    used = [False] * len(detections)
    for i, det in enumerate(detections):
        if used[i]:
            continue
        if det.label not in MERGE_LABELS:
            merged.append(det)
            used[i] = True
            continue

        x1, y1, x2, y2 = det.box
        confidence = det.confidence
        class_id = det.class_id

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            other = detections[j]
            if other.label != det.label:
                continue
            if should_merge_boxes(det.label, (x1, y1, x2, y2), other.box):
                used[j] = True
                x1 = min(x1, other.box[0])
                y1 = min(y1, other.box[1])
                x2 = max(x2, other.box[2])
                y2 = max(y2, other.box[3])
                confidence = max(confidence, other.confidence)

        used[i] = True
        merged.append(Detection(det.label, confidence, (x1, y1, x2, y2), class_id))

    return merged


def _create_kalman_filter(initial: Tuple[float, float, float, float]) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1.0  # cx
    kf.measurementMatrix[1, 1] = 1.0  # cy
    kf.measurementMatrix[2, 4] = 1.0  # w
    kf.measurementMatrix[3, 5] = 1.0  # h
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(8, dtype=np.float32)

    cx, cy, w, h = initial
    state = np.array([[cx], [cy], [0.0], [0.0], [w], [h], [0.0], [0.0]], dtype=np.float32)
    kf.statePost = state.copy()
    kf.statePre = state.copy()
    return kf


def _update_transition(kf: cv2.KalmanFilter, dt: float) -> None:
    kf.transitionMatrix = np.array(
        [
            [1, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


@dataclass
class Track:
    track_id: int
    label: str
    class_id: int
    confidence: float
    box: Tuple[float, float, float, float]
    last_frame: int
    lines_seen: Set[str] = field(default_factory=set)
    chatgpt_lines: Set[str] = field(default_factory=set)
    missed: int = 0
    kalman: cv2.KalmanFilter | None = None

    def predict(self, dt: float) -> Tuple[float, float, float, float]:
        if self.kalman is None:
            return self.box
        _update_transition(self.kalman, dt)
        prediction = self.kalman.predict()
        cx, cy, w, h = (
            float(prediction[0, 0]),
            float(prediction[1, 0]),
            float(prediction[4, 0]),
            float(prediction[5, 0]),
        )
        self.box = cxcywh_to_box(cx, cy, w, h)
        return self.box

    def correct(self, detection: Detection, frame_idx: int) -> None:
        self.label = detection.label
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.last_frame = frame_idx
        self.missed = 0

        if self.kalman is None:
            initial = box_to_cxcywh(detection.box)
            self.kalman = _create_kalman_filter(initial)
            self.box = detection.box
            return

        measurement = np.array([[v] for v in box_to_cxcywh(detection.box)], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        cx, cy, w, h = (
            float(corrected[0, 0]),
            float(corrected[1, 0]),
            float(corrected[4, 0]),
            float(corrected[5, 0]),
        )
        self.box = cxcywh_to_box(cx, cy, w, h)


__all__ = [
    "Track",
    "MERGE_LABELS",
    "box_to_cxcywh",
    "cxcywh_to_box",
    "expand_box",
    "compute_iou",
    "should_merge_boxes",
    "merge_vehicle_detections",
]
