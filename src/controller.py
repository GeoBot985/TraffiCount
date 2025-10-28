"""Controller layer connecting the frontend with the video worker."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from vehicle_dataset import DATASET
from video_processing import extract_frame, read_video_metadata, VideoProcessingError
from capture_store import clear_capture_directory
from video_worker import process_video


stop_event = threading.Event()
worker_thread: Optional[threading.Thread] = None

_FILE_WAIT_TIMEOUT = 10.0
_FILE_WAIT_INTERVAL = 0.2


def _resolve_video_path(video_file) -> Path:
    if video_file is None:
        raise FileNotFoundError("No video selected.")
    if isinstance(video_file, Path):
        path = video_file
    elif hasattr(video_file, "name"):
        path = Path(video_file.name)
    else:
        path = Path(str(video_file))
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at {path}")
    return path


def _wait_for_file_ready(path: Path) -> None:
    """Wait until Gradio finishes writing the uploaded file."""
    deadline = time.monotonic() + _FILE_WAIT_TIMEOUT
    last_size = -1
    while time.monotonic() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                return
            last_size = size
        time.sleep(_FILE_WAIT_INTERVAL)
    if not path.exists():
        raise FileNotFoundError(f"Video file not available: {path}")


def start_job(video_file, overlays_payload=None):
    """Launch the video processing worker and stream annotated frames."""
    try:
        source_path = _resolve_video_path(video_file)
        _wait_for_file_ready(source_path)
    except Exception as exc:
        message = f"Error preparing video: {exc}"
        counts = get_counts()
        return None, message, counts[0], counts[1]

    stop_event.clear()
    DATASET.clear()
    clear_capture_directory()
    status_message = f"Started detection for {source_path.name}"

    frames: list = []

    def on_frame(frame):
        frames[:] = [frame]

    def run_worker():
        try:
            process_video(source_path, on_frame, stop_event, overlays_payload or [])
        except Exception as exc:  # pragma: no cover - guardrail for worker thread
            frames[:] = []
            stop_event.set()
            print(f"[Worker] Error: {exc}")
        finally:
            stop_event.set()

    global worker_thread
    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()

    while not stop_event.is_set():
        if frames:
            detected, identified = get_counts()
            yield frames[-1], status_message, detected, identified
        time.sleep(0.2)

    detected, identified = get_counts()
    yield None, "Detection complete.", detected, identified
    DATASET.clear()


def stop_job():
    """Signal the video worker to stop."""
    stop_event.set()
    detected, identified = get_counts()
    return "Detection stop requested.", detected, identified


def get_counts():
    """Return detection counts for display in the UI."""
    detected = str(DATASET.detected_count())
    identified = str(DATASET.identified_count())
    return detected, identified


def load_frame_preview(video_file):
    """Extract the first frame of the selected video for preview."""
    if not video_file:
        return None, "No file selected."
    try:
        source_path = _resolve_video_path(video_file)
        _wait_for_file_ready(source_path)
        metadata = read_video_metadata(source_path)
        frame = extract_frame(source_path, 0.0, metadata)
        return frame, f"Preview frame 0 (total frames: {metadata.frame_count})"
    except (FileNotFoundError, VideoProcessingError) as exc:
        return None, f"Error: {exc}"
    except Exception as exc:  # pragma: no cover - safety net
        return None, f"Unexpected error: {exc}"
