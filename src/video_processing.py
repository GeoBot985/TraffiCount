"""Video processing utilities for frame extraction, metadata inspection, and detection."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import threading
import time
from obj_detection import Detection, YOLODetector, draw_detections

try:
    import cv2
except ImportError as import_error:
    raise ImportError(
        "OpenCV (cv2) is required for video processing. Install it with `pip install opencv-python`."
    ) from import_error


class VideoProcessingError(RuntimeError):
    """Raised when video metadata or frame extraction fails."""


class VideoProcessingCancelled(VideoProcessingError):
    """Raised when long-running detection is cancelled."""


@dataclass(slots=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int

    def as_payload(self) -> Dict[str, float | int | str]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, float | int | str]) -> "VideoMetadata":
        return cls(
            path=str(payload["path"]),
            fps=float(payload["fps"]),
            frame_count=int(payload["frame_count"]),
            duration=float(payload["duration"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
    
    @classmethod
    def from_file(cls, path: str | Path) -> "VideoMetadata":
        """Reintroduce backward-compatible constructor."""
        from video_processing import read_video_metadata
        return read_video_metadata(path)



@dataclass(slots=True)
class FrameDetections:
    frame_index: int
    timestamp: float
    detections: List[Detection]

    def as_payload(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "detections": [
                {
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "box": detection.box,
                    "class_id": detection.class_id,
                }
                for detection in self.detections
            ],
        }


def _open_capture(video_path: Path, retries: int = 5, delay: float = 0.4) -> cv2.VideoCapture:
    last_error: Optional[str] = None
    for attempt in range(retries):
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            return capture
        last_error = f"attempt {attempt + 1}"
        capture.release()
        time.sleep(delay)
    raise VideoProcessingError(f"Unable to open video: {video_path} ({last_error})")


def read_video_metadata(path: Path | str) -> VideoMetadata:
    video_path = Path(path)
    capture = _open_capture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if fps <= 0:
        raise VideoProcessingError("Video FPS is zero; cannot compute duration.")

    duration = frame_count / fps if frame_count else 0.0
    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        width=width,
        height=height,
    )


def extract_frame(
    path: Path | str,
    seconds: float,
    metadata: Optional[VideoMetadata] = None,
) -> np.ndarray:
    video_path = Path(path)
    info = metadata or read_video_metadata(video_path)
    if info.frame_count == 0:
        raise VideoProcessingError("Video has no frames to extract.")

    target_frame = _seconds_to_frame_index(seconds, info)

    capture = _open_capture(video_path)

    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame_bgr = capture.read()
    capture.release()

    if not success or frame_bgr is None:
        raise VideoProcessingError(f"Failed to read frame {target_frame} from {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


def frame_step_seconds(metadata: VideoMetadata) -> float:
    if metadata.fps <= 0:
        return 1.0
    return max(1.0 / metadata.fps, 0.01)


def detect_objects_at_timestamp(
    path: Path | str,
    seconds: float,
    detector: YOLODetector,
    metadata: Optional[VideoMetadata] = None,
) -> Tuple[np.ndarray, FrameDetections]:
    info = metadata or read_video_metadata(path)
    frame_rgb = extract_frame(path, seconds, info)
    frame_index = _seconds_to_frame_index(seconds, info)
    detections = detector.detect(frame_rgb)
    timestamp = frame_index / info.fps if info.fps > 0 else seconds
    return frame_rgb, FrameDetections(frame_index=frame_index, timestamp=timestamp, detections=detections)


def iter_detections(
    path: Path | str,
    detector: YOLODetector,
    metadata: Optional[VideoMetadata] = None,
    stride_seconds: Optional[float] = None,
) -> Iterator[Tuple[int, float, np.ndarray, List[Detection]]]:
    info = metadata or read_video_metadata(path)
    capture = _open_capture(Path(path))

    fps = info.fps if info.fps > 0 else 30.0
    step_seconds = stride_seconds if stride_seconds is not None else max(1.0 / fps, 0.01)
    step_frames = max(int(round(step_seconds * fps)), 1)

    frame_index = 0
    while frame_index < info.frame_count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame_bgr = capture.read()
        if not success or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame_rgb)
        timestamp = frame_index / fps
        yield frame_index, timestamp, frame_rgb, detections
        frame_index += step_frames

    capture.release()


def annotate_video_with_detections(
    path: Path | str,
    detector: YOLODetector,
    output_path: Path | str | None = None,
    metadata: Optional[VideoMetadata] = None,
    stride_frames: int = 1,
    stop_event: threading.Event | None = None,
    max_frames: Optional[int] = None,
) -> Tuple[Path, List[Detection], float]:
    """Generate a new video with detection boxes drawn on each frame."""

    source_path = Path(path)
    info = metadata or read_video_metadata(source_path)

    capture = _open_capture(source_path)

    fps = info.fps if info.fps > 0 else 30.0
    frame_size = (info.width, info.height)
    if output_path is None:
        output_path = source_path.parent / f"{source_path.stem}_detected.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError(f"Unable to create video writer at: {output_path}")

    last_detections: List[Detection] = []
    detection_snapshot: List[Detection] = []
    detection_timestamp = 0.0
    frame_index = 0
    stride_frames = max(1, int(stride_frames))
    frame_limit = max_frames if max_frames is not None else info.frame_count

    try:
        while frame_index < frame_limit:
            if stop_event is not None and stop_event.is_set():
                raise VideoProcessingCancelled("Detection cancelled by user.")

            success, frame_bgr = capture.read()
            if not success or frame_bgr is None:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if frame_index % stride_frames == 0 or not last_detections:
                last_detections = detector.detect(frame_rgb)

            if last_detections:
                detection_snapshot = last_detections
                detection_timestamp = frame_index / fps if fps > 0 else 0.0

            annotated = draw_detections(frame_rgb, last_detections)
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise VideoProcessingError(f"No frames processed for video: {source_path}")

    if stop_event is not None and stop_event.is_set():
        raise VideoProcessingCancelled("Detection cancelled by user.")

    return output_path, detection_snapshot, detection_timestamp




def _seconds_to_frame_index(seconds: float, metadata: VideoMetadata) -> int:
    clamped = max(0.0, min(seconds, metadata.duration))
    frame = int(round(clamped * metadata.fps))
    return min(max(frame, 0), max(metadata.frame_count - 1, 0))
