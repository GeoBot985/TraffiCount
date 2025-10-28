"""Multi-object tracking utilities with Kalman filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from obj_detection import Detection


@dataclass
class TrackedObject:
    object_id: int
    label: str
    confidence: float
    box: Tuple[float, float, float, float]
    class_id: int


class _KalmanTrack:
    """Represents a single Kalman-filtered track."""

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.id = track_id
        cx, cy = _center_from_box(detection.box)
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(4, dtype=np.float32)
        self.width = detection.box[2] - detection.box[0]
        self.height = detection.box[3] - detection.box[1]
        self.label = detection.label
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.box = detection.box
        self.age = 1
        self.missed = 0

    def predict(self) -> None:
        self.state = _F @ self.state
        self.covariance = _F @ self.covariance @ _F_T + _Q
        self.box = _box_from_state(self.state, self.width, self.height)
        self.age += 1
        self.missed += 1

    def update(self, detection: Detection) -> None:
        measurement = np.array(_center_from_box(detection.box), dtype=np.float32)
        residual = measurement - (_H @ self.state)
        s_matrix = _H @ self.covariance @ _H_T + _R
        kalman_gain = self.covariance @ _H_T @ np.linalg.inv(s_matrix)
        self.state = self.state + kalman_gain @ residual
        identity = np.eye(4, dtype=np.float32)
        self.covariance = (identity - kalman_gain @ _H) @ self.covariance

        measured_w = detection.box[2] - detection.box[0]
        measured_h = detection.box[3] - detection.box[1]
        self.width = 0.6 * measured_w + 0.4 * self.width
        self.height = 0.6 * measured_h + 0.4 * self.height

        self.label = detection.label
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.box = _box_from_state(self.state, self.width, self.height)
        self.missed = 0

    def as_tracked_object(self) -> TrackedObject:
        return TrackedObject(
            object_id=self.id,
            label=self.label,
            confidence=self.confidence,
            box=self.box,
            class_id=self.class_id,
        )


class MultiObjectTracker:
    """Simple tracker that associates detections and smooths positions with Kalman filters."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 15, out_of_frame_margin: float = 20.0, min_new_track_confidence: float = 0.6) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.out_of_frame_margin = out_of_frame_margin
        self.min_new_track_confidence = min_new_track_confidence
        self._tracks: Dict[int, _KalmanTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: Sequence[Detection],
        frame_shape: Tuple[int, int] | None = None,
    ) -> List[TrackedObject]:
        if not self._tracks and not detections:
            return []

        out_of_view: List[int] = []
        for track_id, track in list(self._tracks.items()):
            track.predict()
            if frame_shape and _is_box_outside(track.box, frame_shape, self.out_of_frame_margin):
                out_of_view.append(track_id)

        for track_id in out_of_view:
            self._tracks.pop(track_id, None)

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))

        if track_ids and detections:
            distance_matrix = _build_distance_matrix(self._tracks, detections, track_ids)
            assignments = _greedy_assign(distance_matrix, self.max_distance)

            for track_idx, detection_idx in assignments:
                track_id = track_ids[track_idx]
                detection = detections[detection_idx]
                self._tracks[track_id].update(detection)
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(detection_idx)

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            if track.missed > self.max_missed:
                self._tracks.pop(track_id, None)

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if detection.confidence < self.min_new_track_confidence:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _KalmanTrack(track_id, detection)

        tracked = [track.as_tracked_object() for track in self._tracks.values() if track.missed <= self.max_missed]
        tracked.sort(key=lambda item: item.confidence, reverse=True)
        return tracked


_dt = 1.0
_F = np.array(
    [
        [1.0, 0.0, _dt, 0.0],
        [0.0, 1.0, 0.0, _dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_F_T = _F.T
_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_H_T = _H.T
_Q = np.eye(4, dtype=np.float32) * 1.0
_R = np.eye(2, dtype=np.float32) * 4.0


def _center_from_box(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_from_state(state: np.ndarray, width: float, height: float) -> Tuple[float, float, float, float]:
    cx, cy = state[0], state[1]
    half_w = width / 2.0
    half_h = height / 2.0
    return (
        float(cx - half_w),
        float(cy - half_h),
        float(cx + half_w),
        float(cy + half_h),
    )


def _is_box_outside(
    box: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
    margin: float,
) -> bool:
    height, width = frame_shape
    x1, y1, x2, y2 = box
    return x2 < -margin or y2 < -margin or x1 > width + margin or y1 > height + margin


def _build_distance_matrix(
    tracks: Dict[int, _KalmanTrack],
    detections: Sequence[Detection],
    track_ids: Sequence[int],
) -> np.ndarray:
    centers_detections = np.array([_center_from_box(det.box) for det in detections], dtype=np.float32)
    centers_tracks = np.array([tracks[track_id].state[:2] for track_id in track_ids], dtype=np.float32)
    if centers_tracks.size == 0 or centers_detections.size == 0:
        return np.empty((len(centers_tracks), len(centers_detections)))
    diff = centers_tracks[:, None, :] - centers_detections[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances


def _greedy_assign(distance_matrix: np.ndarray, max_distance: float) -> List[Tuple[int, int]]:
    if distance_matrix.size == 0:
        return []
    assignments: List[Tuple[int, int]] = []
    rows, cols = distance_matrix.shape
    unmatched_rows = set(range(rows))
    unmatched_cols = set(range(cols))

    while unmatched_rows and unmatched_cols:
        best_pair = None
        best_distance = float("inf")
        for r in unmatched_rows:
            for c in unmatched_cols:
                distance = float(distance_matrix[r, c])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (r, c)
        if best_pair is None or best_distance > max_distance:
            break
        r, c = best_pair
        assignments.append((r, c))
        unmatched_rows.remove(r)
        unmatched_cols.remove(c)
    return assignments


__all__ = ["TrackedObject", "MultiObjectTracker"]


