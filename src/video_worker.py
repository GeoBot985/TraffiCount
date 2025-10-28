import cv2
import time
from collections import defaultdict
from itertools import count
from math import hypot
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from draw_overlay import OverlayManager, lines_intersecting_box
from chatgpt_client import reclassify_vehicle
from obj_detection import DEFAULT_MODEL_PATH, Detection, YOLODetector, draw_detections
from vehicle_dataset import DATASET, VehicleRecord
from video_processing import read_video_metadata
from capture_store import (
    CAPTURE_DIR,
    build_capture_stem,
    collect_capture_metadata,
    generate_capture_report,
    write_capture_report,
)
from tracking import (
    Track,
    box_to_cxcywh,
    cxcywh_to_box,
    expand_box,
    merge_vehicle_detections,
    compute_iou,
)

MAX_MISSED_FRAMES = 15
IOU_MATCH_THRESHOLD = 0.3
BOX_EXPANSION_RATIO = 0.15
CENTER_MATCH_THRESHOLD = 90.0
LINE_HIT_COOLDOWN = 1.5
LINE_HIT_DISTANCE = 140.0
CHATGPT_RECHECK_THRESHOLD = 0.8
YOLO_MODEL_NAME = DEFAULT_MODEL_PATH.stem

_track_id_counter = count(1)


def _next_track_id() -> int:
    return next(_track_id_counter)


def _record_from_track(track: Track, timestamp: float) -> VehicleRecord:
    if track.lines_seen:
        line_label = ", ".join(sorted(track.lines_seen))
    else:
        line_label = None
    return VehicleRecord(
        timestamp=timestamp,
        object_id=track.track_id,
        classification=track.label,
        confidence=track.confidence,
        source="yolo",
        line_label=line_label,
    )



def _save_line_capture(
    frame_rgb: np.ndarray,
    track: Track,
    line_label: str,
    timestamp: float,
) -> Optional[Path]:
    if frame_rgb is None or frame_rgb.size == 0:
        return None

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    height, width = frame_rgb.shape[:2]
    expanded = expand_box(track.box, BOX_EXPANSION_RATIO, (height, width))
    x1, y1, x2, y2 = expanded
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    base_name = build_capture_stem(
        timestamp,
        track.label,
        line_label,
        track.confidence,
        source="yolo",
        model=YOLO_MODEL_NAME,
    )
    path = CAPTURE_DIR / f"{base_name}.jpg"
    suffix = 1
    while path.exists():
        path = CAPTURE_DIR / f"{base_name}__dup-{suffix}.jpg"
        suffix += 1

    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return path


def _rename_capture(
    capture_path: Optional[Path],
    timestamp: float,
    classification: str,
    line_label: str,
    confidence: float,
    source: str,
    model: str | None = None,
) -> Optional[Path]:
    if capture_path is None or not capture_path.exists():
        return capture_path

    base_name = build_capture_stem(
        timestamp,
        classification,
        line_label,
        confidence,
        source=source,
        model=model,
    )
    new_path = capture_path.with_name(f"{base_name}.jpg")

    if new_path == capture_path:
        return capture_path

    suffix = 1
    while new_path.exists():
        new_path = capture_path.with_name(f"{base_name}__dup-{suffix}.jpg")
        suffix += 1

    try:
        capture_path.rename(new_path)
    except OSError as exc:
        print(f"[ChatGPT] Failed to rename capture {capture_path} -> {new_path}: {exc}")
        return capture_path
    return new_path


def process_video(video_path, frame_callback, stop_event, overlay_lines=None):
    """
    Run YOLO on every frame, track objects across frames, detect line crossings,
    and push annotated frames to the UI via frame_callback().
    """
    detector = YOLODetector()
    meta = read_video_metadata(video_path)
    cap = cv2.VideoCapture(str(video_path))

    overlay_manager = OverlayManager.from_payload(overlay_lines or [])
    overlay_objects = overlay_manager.lines

    tracks: Dict[int, Track] = {}
    recent_line_hits: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    fps = meta.fps if meta.fps > 0 else 25
    frame_interval = 1.0 / fps
    frame_idx = 0

    while cap.isOpened() and not stop_event.is_set():
        success, frame_bgr = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = list(detector.detect(frame_rgb))
        detections = merge_vehicle_detections(detections)

        timestamp = frame_idx / fps

        for track in tracks.values():
            track.predict(frame_interval)

        # Filter Detection objects only
        valid_detections: List[Detection] = [det for det in detections if isinstance(det, Detection)]

        # --- Associate detections with existing tracks via IoU ---
        unmatched_detections = set(range(len(valid_detections)))
        unmatched_tracks = set(tracks.keys())
        matches: List[Tuple[int, int]] = []

        if valid_detections and tracks:
            iou_pairs: List[Tuple[float, int, int]] = []
            for det_idx, detection in enumerate(valid_detections):
                for track_id, track in tracks.items():
                    iou = compute_iou(detection.box, track.box)
                    if iou >= IOU_MATCH_THRESHOLD:
                        iou_pairs.append((iou, det_idx, track_id))

            for _, det_idx, track_id in sorted(iou_pairs, reverse=True):
                if det_idx in unmatched_detections and track_id in unmatched_tracks:
                    matches.append((det_idx, track_id))
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(track_id)

        if unmatched_detections and unmatched_tracks:
            distance_pairs: List[Tuple[float, int, int]] = []
            for det_idx in unmatched_detections:
                det_cx, det_cy, _, _ = box_to_cxcywh(valid_detections[det_idx].box)
                for track_id in unmatched_tracks:
                    track_cx, track_cy, _, _ = box_to_cxcywh(tracks[track_id].box)
                    dist = hypot(det_cx - track_cx, det_cy - track_cy)
                    if dist <= CENTER_MATCH_THRESHOLD:
                        distance_pairs.append((dist, det_idx, track_id))

            for _, det_idx, track_id in sorted(distance_pairs):
                if det_idx in unmatched_detections and track_id in unmatched_tracks:
                    matches.append((det_idx, track_id))
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(track_id)

        # --- Update matched tracks ---
        for det_idx, track_id in matches:
            detection = valid_detections[det_idx]
            track = tracks[track_id]
            track.correct(detection, frame_idx)

        # --- Handle unmatched tracks ---
        for track_id in list(unmatched_tracks):
            track = tracks[track_id]
            track.missed += 1
            if track.missed > MAX_MISSED_FRAMES:
                del tracks[track_id]

        # --- Create new tracks for unmatched detections ---
        for det_idx in unmatched_detections:
            detection = valid_detections[det_idx]
            track_id = _next_track_id()
            track = Track(
                track_id=track_id,
                label=detection.label,
                class_id=detection.class_id,
                confidence=detection.confidence,
                box=detection.box,
                last_frame=frame_idx,
            )
            tracks[track_id] = track
            track.correct(detection, frame_idx)

        # --- Handle line crossings per track ---
        if overlay_objects:
            for track in tracks.values():
                lines_hit = lines_intersecting_box(overlay_objects, track.box)
                for line_label in lines_hit:
                    if line_label in track.lines_seen:
                        continue
                    hits = recent_line_hits[line_label]
                    hits[:] = [
                        (ts, hx, hy)
                        for ts, hx, hy in hits
                        if timestamp - ts <= LINE_HIT_COOLDOWN
                    ]
                    cx, cy, _, _ = box_to_cxcywh(track.box)
                    if any(hypot(cx - hx, cy - hy) <= LINE_HIT_DISTANCE for ts, hx, hy in hits):
                        continue
                    hits.append((timestamp, cx, cy))
                    track.lines_seen.add(line_label)
                    dataset_record = _record_from_track(track, timestamp)
                    DATASET.add_or_update(dataset_record)
                    print(
                        f"[Stage1] track {track.track_id} ({track.label}) "
                        f"conf={track.confidence:.2f} line={line_label} t={timestamp:.2f}s"
                    )
                    capture_path = _save_line_capture(frame_rgb, track, line_label, timestamp)
                    if (
                        capture_path
                        and track.confidence < CHATGPT_RECHECK_THRESHOLD
                        and line_label not in track.chatgpt_lines
                    ):
                        original_label = track.label
                        original_confidence = track.confidence
                        result = reclassify_vehicle(
                            capture_path,
                            track.label,
                            line_label,
                            track.confidence,
                        )
                        if result is not None:
                            new_label, new_confidence, model_name = result
                            if new_label == original_label:
                                if abs(new_confidence - original_confidence) < 1e-6:
                                    new_confidence = max(
                                        min(original_confidence + 0.2, 0.95),
                                        original_confidence,
                                    )
                                else:
                                    new_confidence = (new_confidence + original_confidence) / 2.0
                            capture_path = _rename_capture(
                                capture_path,
                                timestamp,
                                new_label,
                                line_label,
                                new_confidence,
                                source="chatgpt",
                                model=model_name,
                            )
                            track.label = new_label
                            track.confidence = new_confidence
                            track.chatgpt_lines.add(line_label)
                            chatgpt_record = VehicleRecord(
                                timestamp=timestamp,
                                object_id=track.track_id,
                                classification=new_label,
                                confidence=new_confidence,
                                source="chatgpt",
                                line_label=line_label,
                            )
                            DATASET.add_or_update(chatgpt_record)
                            print(
                                f"[ChatGPT] track {track.track_id} "
                                f"{original_label}->{new_label} "
                                f"conf {original_confidence:.2f}->{new_confidence:.2f} "
                                f"line={line_label} t={timestamp:.2f}s model={model_name}"
                            )

        # --- Draw boxes with track identifiers for visualization ---
        annotated_detections: List[Detection] = []
        for track in tracks.values():
            label = f"{track.label}#{track.track_id}"
            expanded_box = expand_box(track.box, BOX_EXPANSION_RATIO, frame_rgb.shape[:2])
            annotated_detections.append(
                Detection(
                    label=label,
                    confidence=track.confidence,
                    box=expanded_box,
                    class_id=track.class_id,
                )
            )

        annotated = draw_detections(frame_rgb, annotated_detections)

        # --- Send frame to frontend if callback provided ---
        if callable(frame_callback):
            frame_callback(annotated)

        frame_idx += 1

        detections_present = len(valid_detections)
        active_tracks = len(tracks)
        if detections_present == 0 and active_tracks == 0:
            sleep_time = max(0.005, frame_interval * 0.25)
        elif detections_present <= 1 and active_tracks <= 1:
            sleep_time = max(0.005, frame_interval * 0.5)
        else:
            sleep_time = frame_interval
        time.sleep(sleep_time)

    cap.release()
    records = collect_capture_metadata()
    report_lines = generate_capture_report(records)
    report_path = write_capture_report(report_lines)
    if records:
        print(f"[Report] Wrote capture summary to {report_path}")
    else:
        print(f"[Report] No captures found; wrote empty report to {report_path}")
    return
