"""Utilities for managing and rendering overlay lines on video frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Any

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as import_error:
    raise ImportError(
        "Pillow is required for overlay rendering. Install it with `pip install pillow`."
    ) from import_error

Point = Tuple[int, int]
SerializedPoint = Tuple[int, int]
SerializedLine = Dict[str, Any]

_COLOR_PALETTE = (
    "#FF6B6B",
    "#4ECDC4",
    "#FFD93D",
    "#1A535C",
    "#9368B7",
    "#F25F5C",
)


@dataclass
class OverlayLine:
    """Represents a single labeled overlay line."""

    label: str
    points: Tuple[Point, Point]

    def to_payload(self) -> SerializedLine:
        return {
            "label": self.label,
            "points": [
                (int(self.points[0][0]), int(self.points[0][1])),
                (int(self.points[1][0]), int(self.points[1][1])),
            ],
        }

    @classmethod
    def from_payload(cls, payload: SerializedLine) -> "OverlayLine":
        label = str(payload["label"])
        raw_points = payload.get("points")
        if not isinstance(raw_points, Sequence) or len(raw_points) != 2:
            raise ValueError("Overlay lines require exactly two points.")
        p1 = _coerce_point(raw_points[0])
        p2 = _coerce_point(raw_points[1])
        return cls(label=label, points=(p1, p2))

    def center(self) -> Point:
        (x1, y1), (x2, y2) = self.points
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class OverlayManager:
    """Tracks overlay lines for a single frame or video."""

    def __init__(self, lines: Iterable[OverlayLine] | None = None) -> None:
        self._lines: List[OverlayLine] = list(lines) if lines else []

    def add_line(self, label: str, points: Sequence[Point]) -> OverlayLine:
        if len(points) != 2:
            raise ValueError("Overlay lines must have exactly two points.")
        normalized_points = (_coerce_point(points[0]), _coerce_point(points[1]))
        if not label:
            raise ValueError("Overlay lines require a label.")
        if any(existing.label == label for existing in self._lines):
            raise ValueError(f"Duplicate overlay label: {label}")
        line = OverlayLine(label=label, points=normalized_points)
        self._lines.append(line)
        return line

    def remove_line(self, label: str) -> None:
        self._lines = [line for line in self._lines if line.label != label]

    def clear(self) -> None:
        self._lines.clear()

    @property
    def lines(self) -> List[OverlayLine]:
        return list(self._lines)

    def to_payload(self) -> List[SerializedLine]:
        return [line.to_payload() for line in self._lines]

    @classmethod
    def from_payload(cls, payload: Iterable[SerializedLine] | None) -> "OverlayManager":
        if not payload:
            return cls()
        lines = [OverlayLine.from_payload(item) for item in payload]
        return cls(lines)


def render_overlay_preview(
    frame_rgb: np.ndarray,
    overlays_payload: Iterable[SerializedLine] | None,
    pending_points: Sequence[Point] | None = None,
) -> np.ndarray:
    """Render overlay lines and optionally pending points onto a frame copy."""

    if frame_rgb is None:
        raise ValueError("A base frame is required to render overlays.")

    image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
    drawing = ImageDraw.Draw(image, "RGBA")

    manager = OverlayManager.from_payload(overlays_payload)
    for idx, line in enumerate(manager.lines):
        color = _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]
        drawing.line(line.points, fill=color, width=4)
        _draw_label(drawing, line, color)

    if pending_points:
        normalized = [_coerce_point(point) for point in pending_points[-2:]]
        if len(normalized) == 1:
            _draw_pending_point(drawing, normalized[0])
        elif len(normalized) == 2:
            drawing.line(normalized, fill="#FFFFFF", width=2, joint="curve")
            _draw_pending_point(drawing, normalized[0])
            _draw_pending_point(drawing, normalized[1])

    return np.array(image)


def overlays_table_payload(overlays_payload: Iterable[SerializedLine]) -> List[List[str]]:
    rows: List[List[str]] = []
    for entry in overlays_payload or []:
        line = OverlayLine.from_payload(entry)
        (x1, y1), (x2, y2) = line.points
        rows.append([line.label, f"({x1}, {y1}) -> ({x2}, {y2})"])
    return rows


def _coerce_point(point: Sequence[Any]) -> Point:
    if len(point) < 2:
        raise ValueError("Points must contain an x and y coordinate.")
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _draw_label(drawing: ImageDraw.ImageDraw, line: OverlayLine, color: str) -> None:
    text = line.label
    center = line.center()
    bbox = drawing.textbbox(center, text, anchor="mm")
    if bbox:
        x0, y0, x1, y1 = bbox
        padding = 4
        background = (0, 0, 0, 180)
        drawing.rectangle(
            (x0 - padding, y0 - padding, x1 + padding, y1 + padding),
            fill=background,
            outline=color,
            width=1,
        )
    drawing.text(center, text, fill="#FFFFFF", anchor="mm")




def lines_intersecting_box(lines: Sequence[OverlayLine], box: Tuple[float, float, float, float]) -> List[str]:
    if not lines:
        return []
    return [line.label for line in lines if _line_intersects_box(line.points, box)]


def _line_intersects_box(points: Tuple[Point, Point], box: Tuple[float, float, float, float]) -> bool:
    p1, p2 = points
    if _point_in_rect(p1, box) or _point_in_rect(p2, box):
        return True

    x1, y1, x2, y2 = box
    edges = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for edge_start, edge_end in edges:
        if _segments_intersect(p1, p2, edge_start, edge_end):
            return True
    return False


def _point_in_rect(point: Point, box: Tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _segments_intersect(
    p1: Point,
    p2: Point,
    q1: Point,
    q2: Point,
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True
    return False


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= 1e-6:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) - 1e-6 <= b[0] <= max(a[0], c[0]) + 1e-6
        and min(a[1], c[1]) - 1e-6 <= b[1] <= max(a[1], c[1]) + 1e-6
    )

def _draw_pending_point(drawing: ImageDraw.ImageDraw, point: Point) -> None:
    radius = 6
    x, y = point
    drawing.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#FFFFFF", width=2)
