
# ===========================================================================
# FILE START: src\atest.py
# ===========================================================================

from video_processing import extract_frame, read_video_metadata
from pathlib import Path

path = Path("testVid1.mp4")
meta = read_video_metadata(path)
print(meta)

frame = extract_frame(path, 0)
print("Frame:", type(frame), frame.shape if frame is not None else None)


# ===========================================================================
# FILE START: src\capture_store.py
# ===========================================================================

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CAPTURE_DIR = Path("data/captures")


def _sanitize_part(value: str, allow_dot: bool = False) -> str:
    allowed = {"-", "_"}
    if allow_dot:
        allowed.add(".")
    return "".join(ch if ch.isalnum() or ch in allowed else "_" for ch in value)


def build_capture_stem(
    timestamp: float,
    classification: str,
    line_label: str,
    confidence: float,
    source: str,
    model: str | None = None,
) -> str:
    seconds_total = max(0, timestamp)
    minutes = int(seconds_total // 60)
    seconds = int(round(seconds_total % 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    ts_text = f"{str(minutes).zfill(2)}{seconds:02d}"
    parts = [
        ("ts", ts_text),
        ("veh", _sanitize_part(classification or "vehicle")),
        ("line", _sanitize_part(line_label or "line")),
        ("conf", _sanitize_part(f"{confidence:.2f}", allow_dot=True)),
        ("src", _sanitize_part(source or "unknown")),
    ]
    if model:
        parts.append(("model", _sanitize_part(model)))
    return "__".join(f"{key}-{value}" for key, value in parts)


def parse_capture_metadata(path: Path) -> Optional[Dict[str, Any]]:
    stem = path.stem
    parts = stem.split("__")
    data: Dict[str, str] = {}
    for part in parts:
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        if key == "dup":
            continue
        data[key] = value
    required_keys = {"ts", "veh", "line", "conf", "src"}
    if not required_keys.issubset(data):
        return None
    ts_raw = data["ts"]
    if len(ts_raw) < 2:
        return None
    minutes_part = ts_raw[:-2] or "0"
    seconds_part = ts_raw[-2:]
    try:
        minutes = int(minutes_part)
        seconds = int(seconds_part)
    except ValueError:
        return None
    timestamp = max(0.0, minutes * 60 + seconds)
    confidence_str = data["conf"].replace("_", ".")
    try:
        confidence = float(confidence_str)
    except ValueError:
        confidence = 0.0

    def _restore(value: str) -> str:
        return value.replace("_", " ").strip()

    return {
        "timestamp": timestamp,
        "vehicle": _restore(data["veh"]),
        "line": _restore(data["line"]),
        "confidence": confidence,
        "source": _restore(data["src"]),
        "model": _restore(data.get("model", "")),
        "path": path,
    }


def generate_capture_report(records: Iterable[Dict[str, Any]], interval_s: int = 900) -> List[str]:

    def bracket_label(ts: float) -> str:
        bucket = int(ts // interval_s)
        start_min = bucket * (interval_s // 60)
        end_min = start_min + (interval_s // 60)
        return f"{start_min:02d}-{end_min:02d}min"

    summary: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        bracket = bracket_label(record["timestamp"])
        line = record["line"] or "-"
        key = (bracket, line)
        bucket_entry = summary.setdefault(
            key,
            {"total": 0, "chatgpt": 0, "types": defaultdict(int)},
        )
        bucket_entry["total"] += 1
        vehicle = record["vehicle"] or "unknown"
        bucket_entry["types"][vehicle] += 1
        source = record["source"].lower()
        if source.startswith("chatgpt"):
            bucket_entry["chatgpt"] += 1

    lines: List[str] = ["Vehicle Capture Summary", "=======================", ""]
    for (bracket, line_label) in sorted(summary):
        entry = summary[(bracket, line_label)]
        lines.append(f"Time Bracket: {bracket}, Line: {line_label}")
        lines.append(f"  total vehicles: {entry['total']}")
        lines.append(f"  chatgpt referrals: {entry['chatgpt']}")
        for vehicle, count in sorted(entry["types"].items()):
            lines.append(f"  {vehicle}: {count}")
        lines.append("")
    return lines


def write_capture_report(lines: List[str], directory: Path = CAPTURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.txt"
    if lines:
        report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    else:
        report_path.write_text("Vehicle Capture Summary\n\nNo captures recorded.\n", encoding="utf-8")
    return report_path


def collect_capture_metadata(directory: Path = CAPTURE_DIR) -> List[Dict[str, Any]]:
    if not directory.exists():
        return []
    records: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.jpg")):
        meta = parse_capture_metadata(path)
        if meta is not None:
            records.append(meta)
    return records


def clear_capture_directory(directory: Path = CAPTURE_DIR) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.jpg"):
        try:
            path.unlink()
        except OSError as exc:
            print(f"[Cleanup] Failed to remove {path}: {exc}")
    report_path = directory / "report.txt"
    if report_path.exists():
        try:
            report_path.unlink()
        except OSError as exc:
            print(f"[Cleanup] Failed to remove {report_path}: {exc}")


__all__ = [
    "CAPTURE_DIR",
    "build_capture_stem",
    "parse_capture_metadata",
    "collect_capture_metadata",
    "generate_capture_report",
    "write_capture_report",
    "clear_capture_directory",
]



# ===========================================================================
# FILE START: src\chatgpt_client.py
# ===========================================================================

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency might be optional in some environments
    OpenAI = None  # type: ignore[assignment]

API_KEY_PATH = Path("src/api_key.txt")
_client: OpenAI | None = None


@lru_cache(maxsize=1)
def _load_api_key() -> Optional[str]:
    if not API_KEY_PATH.exists():
        print(f"[ChatGPT] API key file missing at {API_KEY_PATH}")
        return None
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        print("[ChatGPT] API key file is empty")
        return None
    return key


def _get_client() -> Optional[OpenAI]:
    global _client
    if OpenAI is None:
        print("[ChatGPT] openai package not installed; skipping reclassification.")
        return None
    if _client is None:
        api_key = _load_api_key()
        if not api_key:
            return None
        _client = OpenAI(api_key=api_key)
    return _client


def _collect_text(response: object) -> str:
    """
    Extract concatenated text segments from a Responses API payload.
    Fall back to string conversion if structure is unexpected.
    """
    def _strip_code_fence(text: str) -> str:
        trimmed = text.strip()
        if trimmed.startswith("```"):
            without_lead = trimmed[3:]
            if "\n" in without_lead:
                _, remainder = without_lead.split("\n", 1)
            else:
                remainder = without_lead
            if remainder.endswith("```"):
                remainder = remainder[:-3]
            return remainder.strip()
        return trimmed

    try:
        output_items = getattr(response, "output", None)
        if not output_items:
            candidates = getattr(response, "data", None)
            if isinstance(candidates, list):
                return "".join(
                    _strip_code_fence(
                        getattr(choice, "text", "") or getattr(choice, "content", "") or ""
                    )
                    for choice in candidates
                )
            return str(response)
        parts: list[str] = []
        for item in output_items:
            contents = getattr(item, "content", [])
            for content in contents:
                if getattr(content, "type", None) == "output_text":
                    parts.append(_strip_code_fence(getattr(content, "text", "")))
        return "".join(parts)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ChatGPT] Failed to collect text from response: {exc}")
        return str(response)


def reclassify_vehicle(
    image_path: Path,
    current_label: str,
    line_label: Optional[str],
    current_confidence: float,
) -> Optional[Tuple[str, float, str]]:
    """
    Send a captured image to GPT for reclassification.

    Returns a tuple of (classification, confidence, model_used) if successful.
    """
    client = _get_client()
    if client is None:
        return None

    if not image_path.exists():
        print(f"[ChatGPT] Image not found for reclassification: {image_path}")
        return None

    with image_path.open("rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("utf-8")

    line_text = line_label or "unknown line"
    # Spell out the business taxonomy so the model never guesses outside our allowed labels.
    prompt = (
        "You are an expert at identifying vehicles from traffic camera still images. "
        "Apply the following taxonomy strictly:\n"
        "  • Any passenger car, SUV, pickup/bakkie, minivan, or delivery van built on a light chassis "
        "must be labeled 'light vehicle'.\n"
        "  • Only heavy goods vehicles (multi-axle trucks, large box trucks, articulated lorries) "
        "may be labeled 'truck'.\n"
        "  • Only label 'taxi' if the vehicle is a Toyota Quantum minibus or shows unmistakable taxi signage "
        "(a roof light, a 'TAXI' decal, the South African yellow lateral stripe, or clear commuter markings). "
        "If these cues are absent, do NOT guess taxi—fall back to 'light vehicle'.\n"
        "  • When the body resembles a van or pickup truck without explicit taxi markings, treat it as a light vehicle.\n"
        "If the evidence is ambiguous, prefer 'light vehicle' over 'taxi'.\n"
        f"Our current computer vision classification is '{current_label}' with "
        f"confidence {current_confidence:.2f} for line '{line_text}'. "
        "Respond with raw JSON (no code fences) containing keys 'classification' "
        "(lowercase string), 'confidence' (float between 0 and 1), and 'model' "
        "(short string identifying the model you used)."
    )

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": "You answer using strict JSON."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:
        print(f"[ChatGPT] API request failed: {exc}")
        return None

    raw_text = _collect_text(response).strip()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[ChatGPT] Failed to parse JSON response '{raw_text}': {exc}")
        return None

    classification = data.get("classification")
    new_confidence = data.get("confidence")
    model_used = data.get("model") or "gpt-4o-mini"

    if not isinstance(classification, str):
        print(f"[ChatGPT] Invalid classification in response: {data}")
        return None
    try:
        confidence_value = float(new_confidence)
    except (TypeError, ValueError):
        print(f"[ChatGPT] Invalid confidence in response: {data}")
        return None

    confidence_value = max(0.0, min(1.0, confidence_value))
    return classification.strip().lower(), confidence_value, str(model_used)


__all__ = ["reclassify_vehicle"]



# ===========================================================================
# FILE START: src\controller.py
# ===========================================================================

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



# ===========================================================================
# FILE START: src\download_obj_detector.py
# ===========================================================================

"""Download helper for object detector weights (YOLO models)."""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    repo_id: str
    filename: str
    sha256: str
    exported_format: Optional[str] = "onnx"
    exported_name: Optional[str] = None
    requires_auth: bool = False

    @property
    def target_filename(self) -> str:
        if not self.exported_format:
            return self.filename
        return self.exported_name or f"{self.name}.{self.exported_format}"

    @property
    def download_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"


MODEL_REGISTRY: Dict[str, ModelArtifact] = {
    "yolo11n": ModelArtifact(
        name="yolo11n",
        repo_id="Ultralytics/YOLO11",
        filename="yolo11n.pt",
        sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        exported_format="onnx",
        exported_name="yolo11n.onnx",
        requires_auth=True,
    ),
    "yolo11s": ModelArtifact(
        name="yolo11s",
        repo_id="Ultralytics/YOLO11",
        filename="yolo11s.pt",
        sha256="85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        exported_format="onnx",
        exported_name="yolo11s.onnx",
        requires_auth=True,
    ),
    "yolov8n": ModelArtifact(
        name="yolov8n",
        repo_id="Ultralytics/YOLOv8",
        filename="yolov8n.pt",
        sha256="31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95",
        exported_format="onnx",
        exported_name="yolov8n.onnx",
        requires_auth=True,
    ),
}

DEFAULT_MODEL = "yolo11s"

DEFAULT_OUTPUT_DIR = Path("data/models")
CHUNK_SIZE = 2 ** 20  # 1 MiB
USER_AGENT = "TraffiCount/0.1"
ENV_FILE = Path(".env")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YOLO object detector weights and export to ONNX.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        help="Model identifier to download (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store downloaded weights (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-export even if artifacts already exist.",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        type=str,
        help="Hugging Face token (falls back to HF_TOKEN env var or .env).",
    )
    parser.add_argument(
        "--no-export",
        dest="export",
        action="store_false",
        help="Skip ONNX export and keep only downloaded weights.",
    )
    return parser.parse_args(argv)


def load_env_token() -> Optional[str]:
    if not ENV_FILE.exists():
        return None
    try:
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip()
                if value.startswith("\"") and value.endswith("\""):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                return value
    except OSError as exc:
        print(f"Warning: could not read {ENV_FILE}: {exc}")
    return None


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, headers: Optional[Dict[str, str]] = None) -> None:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request) as response, destination.open("wb") as target:
        total = response.length or 0
        read = 0
        while True:
            block = response.read(CHUNK_SIZE)
            if not block:
                break
            target.write(block)
            read += len(block)
            if total:
                percent = read / total * 100
                print(f"\rDownloading {destination.name}: {percent:.1f}%", end="", flush=True)
        if total:
            print("")


def ensure_download(artifact: ModelArtifact, target_dir: Path, force: bool, headers: Dict[str, str]) -> Path:
    weights_path = target_dir / artifact.filename
    if weights_path.exists() and not force:
        checksum = compute_sha256(weights_path)
        if checksum == artifact.sha256:
            print(f"OK: {artifact.filename} already present (checksum ok).")
            return weights_path
        print(
            f"Warning: checksum mismatch for {weights_path} (expected {artifact.sha256}, got {checksum})."
        )
        weights_path.unlink(missing_ok=True)

    print(f"Fetching {artifact.filename} -> {weights_path}")
    download_file(artifact.download_url, weights_path, headers=headers)
    checksum = compute_sha256(weights_path)
    if checksum != artifact.sha256:
        weights_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Checksum verification failed after download. "
            f"Expected {artifact.sha256}, got {checksum}."
        )
    print(f"OK: downloaded {artifact.filename} (sha256: {checksum[:12]}...)")
    return weights_path


def export_to_onnx(weights_path: Path, output_path: Path, force: bool) -> Path:
    if output_path.exists() and not force:
        print(f"OK: ONNX artifact already exists at {output_path}.")
        return output_path

    if YOLO is None:
        raise RuntimeError(
            "ultralytics package is required for ONNX export. Install it via `pip install ultralytics`."
        )

    print(f"Exporting {weights_path.name} -> {output_path.name} (ONNX)")
    export_result = YOLO(str(weights_path)).export(
        format="onnx",
        dynamic=True,
        simplify=True,
        imgsz=640,
        opset=12,
        device="cpu",
    )

    exported_path: Optional[Path] = None
    if isinstance(export_result, (list, tuple)) and export_result:
        exported_path = Path(export_result[0])
    elif isinstance(export_result, dict) and "model" in export_result:
        exported_path = Path(export_result["model"])
    elif isinstance(export_result, str):
        exported_path = Path(export_result)

    candidate = exported_path or output_path
    if not candidate.exists():
        candidate = weights_path.with_suffix(".onnx")
    if not candidate.exists():
        raise RuntimeError("Exporter did not produce an ONNX file as expected.")

    if candidate != output_path:
        output_path.write_bytes(candidate.read_bytes())
        candidate.unlink(missing_ok=True)

    print(f"OK: export complete at {output_path}")
    return output_path
    if YOLO is None:
        raise RuntimeError(
            "ultralytics package is required for ONNX export. Install it via `pip install ultralytics`."
        )

    print(f"Exporting {weights_path.name} -> {output_path.name} (ONNX)")
    model = YOLO(str(weights_path))
    export_result = model.export(
        format="onnx",
        dynamic=True,
        simplify=True,
        imgsz=640,
        opset=12,
        device="cpu",
    )

    exported_path = Path(export_result) if isinstance(export_result, str) else output_path
    if exported_path != output_path:
        # Move/rename to expected location
        exported_path = Path(export_result)
        output_path.write_bytes(exported_path.read_bytes())
        Path(export_result).unlink(missing_ok=True)
    print(f"OK: export complete at {output_path}")
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    artifact = MODEL_REGISTRY[args.model]
    ensure_directory(args.output_dir)

    token = args.hf_token or os.getenv("HF_TOKEN") or load_env_token()
    headers: Dict[str, str] = {"User-Agent": USER_AGENT}
    if artifact.requires_auth:
        if not token:
            print(
                "Error: Hugging Face token required. Provide --hf-token, set HF_TOKEN, or populate .env."
            )
            return 2
        headers["Authorization"] = f"Bearer {token}"

    try:
        weights_path = ensure_download(artifact, args.output_dir, args.force, headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("Error: authentication failed (401). Check your Hugging Face token permissions.")
            return 1
        if exc.code == 404:
            print("Error: weights not found at remote location (HTTP 404). Validate registry configuration.")
            return 1
        print(f"Error: HTTP {exc.code} {exc.reason} while downloading weights.")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: download failed - {exc}")
        return 1

    if not args.export or not artifact.exported_format:
        print(f"Result: weights stored at {weights_path}")
        return 0

    output_path = args.output_dir / artifact.target_filename
    try:
        export_to_onnx(weights_path, output_path, args.force)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: export failed - {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





# ===========================================================================
# FILE START: src\draw_overlay.py
# ===========================================================================

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



# ===========================================================================
# FILE START: src\file_handler.py
# ===========================================================================

﻿"""File handling helpers for TraffiCount."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class VideoAsset:
    path: Path

    @property
    def processing_root(self) -> Path:
        return self.path.parent / f"{self.path.stem}_processing"


class VideoFileManager:
    """Resolves uploaded or local video references to filesystem paths."""

    def __init__(
        self,
        base_directory: Path | None = None,
        max_wait_seconds: float = 15.0,
        poll_interval: float = 0.25,
        stable_iterations: int = 2,
    ) -> None:
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval = poll_interval
        self.stable_iterations = stable_iterations

    def resolve(self, file_reference: Any) -> VideoAsset:
        path = self._normalize_reference(file_reference)
        expected_size = self._extract_expected_size(file_reference)
        self._wait_for_file_ready(path, expected_size)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return VideoAsset(path=path)

    def ensure_processing_directory(self, asset: VideoAsset) -> Path:
        processing_dir = asset.processing_root
        processing_dir.mkdir(parents=True, exist_ok=True)
        return processing_dir

    def _normalize_reference(self, file_reference: Any) -> Path:
        if isinstance(file_reference, Path):
            return file_reference
        if isinstance(file_reference, str) and file_reference:
            return Path(file_reference)
        if isinstance(file_reference, dict):
            name = file_reference.get("path") or file_reference.get("name")
            if name:
                return Path(name)
        if hasattr(file_reference, "name"):
            return Path(getattr(file_reference, "name"))
        raise ValueError("Unsupported file reference received; expected path-like input.")

    def _extract_expected_size(self, file_reference: Any) -> Optional[int]:
        if isinstance(file_reference, dict):
            for key in ("size", "orig_size", "file_size"):
                size = self._coerce_size(file_reference.get(key))
                if size is not None:
                    return size
        if hasattr(file_reference, "size"):
            size = self._coerce_size(getattr(file_reference, "size"))
            if size is not None:
                return size
        return None

    @staticmethod
    def _coerce_size(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            size = int(value)
            return size if size >= 0 else None
        except (TypeError, ValueError):
            return None

    def _wait_for_file_ready(self, path: Path, expected_size: Optional[int]) -> None:
        if not expected_size and not path.exists():
            return

        deadline = time.monotonic() + self.max_wait_seconds
        last_size = -1
        stable_hits = 0

        while time.monotonic() < deadline:
            if path.exists():
                size = path.stat().st_size
                if expected_size and size >= expected_size:
                    stable_hits += 1
                    if stable_hits >= self.stable_iterations:
                        return
                elif size == last_size and size > 0:
                    stable_hits += 1
                    if stable_hits >= self.stable_iterations:
                        return
                else:
                    stable_hits = 0
                    last_size = size
            time.sleep(self.poll_interval)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found after waiting: {path}")
        if expected_size and path.stat().st_size < expected_size:
            raise TimeoutError(
                f"Video upload appears incomplete for {path} (expected {expected_size} bytes, got {path.stat().st_size})."
            )



# ===========================================================================
# FILE START: src\frontend.py
# ===========================================================================

"""Gradio frontend for TraffiCount."""
from __future__ import annotations

from typing import List

import gradio as gr

from controller import get_counts, load_frame_preview, start_job, stop_job
from draw_overlay import OverlayManager, overlays_table_payload, render_overlay_preview


def _render_preview(base_image, overlays_payload, pending_points=None):
    if base_image is None:
        return None
    try:
        return render_overlay_preview(
            base_image,
            overlays_payload or [],
            pending_points or [],
        )
    except ValueError:
        return base_image


def handle_point_selection(evt: gr.SelectData, overlays_payload, pending_points, current_image):
    """Collect two points for a candidate overlay line."""
    if evt is None or evt.index is None:
        return pending_points, current_image, gr.update(), "Click on the image to set points."

    x, y = map(int, evt.index)
    points = list(pending_points or [])
    points.append((x, y))

    message = f"Point {len(points)} selected at ({x}, {y})."
    updated_image = _render_preview(current_image, overlays_payload, points)
    return points[:2], updated_image, gr.update(), message


def save_line(line_name: str, overlays_payload, pending_points, current_image):
    """Persist a line drawn by the user."""
    if not line_name:
        return overlays_payload, pending_points, current_image, gr.update(), "Enter a line name first."
    if not pending_points or len(pending_points) < 2:
        return overlays_payload, pending_points, current_image, gr.update(), "Select two points before saving."

    manager = OverlayManager.from_payload(overlays_payload or [])
    try:
        manager.add_line(line_name, pending_points[:2])
    except ValueError as exc:
        return overlays_payload, pending_points, current_image, gr.update(), str(exc)

    overlays = manager.to_payload()
    table_rows = overlays_table_payload(overlays)
    updated_image = _render_preview(current_image, overlays, [])
    return overlays, [], updated_image, table_rows, f"Saved line '{line_name}'."


def reset_pending_points(overlays_payload, current_image):
    """Clear in-progress overlay points."""
    updated_image = _render_preview(current_image, overlays_payload, [])
    return [], updated_image, gr.update(value=""), "Pending points cleared."


def clear_overlay_lines(current_image):
    """Remove all saved overlay lines."""
    overlays = []
    table_rows: List[List[str]] = []
    updated_image = _render_preview(current_image, overlays, [])
    return overlays, updated_image, table_rows, [], gr.update(value=""), "All overlay lines cleared."


def reset_after_completion():
    """Reset overlay state and status after a video completes."""
    return [], [], [], gr.update(value=""), "Ready for next video."


def build_frontend():
    with gr.Blocks(title="TraffiCount") as demo:
        gr.Markdown("# TraffiCount")

        overlays_state = gr.State([])
        pending_points_state = gr.State([])

        video_input = gr.File(label="Video file", file_count="single", file_types=[".mp4"])
        detection_view = gr.Image(label="Detection View", image_mode="RGB", type="numpy")
        overlay_table = gr.Dataframe(headers=["Label", "Points"], interactive=False, label="Overlay lines")
        detected_label = gr.Textbox(label="Vehicles Detected", interactive=False, value="0")
        identified_label = gr.Textbox(label="Vehicles Identified", interactive=False, value="0")
        status = gr.Markdown("Ready.")

        with gr.Row():
            start_btn = gr.Button("Start Detection", variant="primary")
            stop_btn = gr.Button("Stop Detection", variant="secondary")
            frame_btn = gr.Button("Load Preview Frame")

        with gr.Row():
            line_name = gr.Textbox(label="Line Name", placeholder="Example: Entry Line A")
            save_line_btn = gr.Button("Save Line")
            reset_points_btn = gr.Button("Reset Points")
            clear_lines_btn = gr.Button("Clear All Lines")

        def refresh_counts():
            return get_counts()

        refresh_btn = gr.Button("Refresh Counts", visible=False)
        refresh_btn.click(refresh_counts, outputs=[detected_label, identified_label])

        start_chain = start_btn.click(
            start_job,
            inputs=[video_input, overlays_state],
            outputs=[detection_view, status, detected_label, identified_label],
        )
        start_chain.then(
            reset_after_completion,
            outputs=[overlays_state, pending_points_state, overlay_table, line_name, status],
        )
        stop_chain = stop_btn.click(stop_job, None, [status, detected_label, identified_label])
        stop_chain.then(
            reset_after_completion,
            outputs=[overlays_state, pending_points_state, overlay_table, line_name, status],
        )

        frame_btn.click(
            load_frame_preview,
            inputs=[video_input],
            outputs=[detection_view, status],
        )

        video_input.change(
            load_frame_preview,
            inputs=[video_input],
            outputs=[detection_view, status],
        )

        detection_view.select(
            handle_point_selection,
            inputs=[overlays_state, pending_points_state, detection_view],
            outputs=[pending_points_state, detection_view, overlay_table, status],
        )

        save_line_btn.click(
            save_line,
            inputs=[line_name, overlays_state, pending_points_state, detection_view],
            outputs=[overlays_state, pending_points_state, detection_view, overlay_table, status],
        )

        reset_points_btn.click(
            reset_pending_points,
            inputs=[overlays_state, detection_view],
            outputs=[pending_points_state, detection_view, line_name, status],
        )

        clear_lines_btn.click(
            clear_overlay_lines,
            inputs=[detection_view],
            outputs=[overlays_state, detection_view, overlay_table, pending_points_state, line_name, status],
        )

        def auto_refresh():
            while True:
                yield get_counts()

        try:
            gr.Timer(2.0, fn=auto_refresh, outputs=[detected_label, identified_label])
        except Exception:
            pass

    return demo


def launch():
    app = build_frontend()
    app.queue()
    app.launch(share=False, show_error=True)


if __name__ == "__main__":
    launch()



# ===========================================================================
# FILE START: src\main.py
# ===========================================================================

﻿from frontend import launch


if __name__ == "__main__":
    launch()



# ===========================================================================
# FILE START: src\obj_detection.py
# ===========================================================================

"""Object detection utilities powered by YOLO ONNX models and ONNX Runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import platform
import cv2
import numpy as np
import onnxruntime as ort


COCO_CLASS_NAMES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

VEHICLE_CLASS_NAMES: Tuple[str, ...] = (
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
)

DEFAULT_MODEL_PATH = Path("data/models/yolo11s.onnx")
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_INPUT_SIZE = 640
# Confidence calibration constants for YOLO outputs. These help align raw detections with
# the downstream review thresholds (e.g. 0.8 for automatic hand-off to GPT).
BUS_CONFIDENCE_SCALE = 0.85          # Nudge bus scores down; real buses are rare so we favour second-pass review.
TRUCK_CONFIDENCE_SCALE = 0.8         # Penalise ambiguous truck boxes (often bakkies/pickups misclassified as trucks).
TRUCK_LARGE_MIN_AREA = 0.08          # If a truck box covers >=8% of the frame, treat it as a heavy vehicle.
TRUCK_LARGE_MIN_ASPECT = 2.4         # Very wide aspect ratios (articulated tankers) should stay confidently "truck".
MIN_VEHICLE_AREA_RATIO = 5e-4        # Ignore detections smaller than 0.05% of the frame (likely pedestrians/noise).
MIN_VEHICLE_HEIGHT_RATIO = 0.05      # Likewise skip boxes shorter than 5% of the frame height.


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]
    class_id: int


@dataclass(slots=True)
class DetectorConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    input_size: int = DEFAULT_INPUT_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    class_filter: Optional[Sequence[str]] = VEHICLE_CLASS_NAMES
    providers: Optional[Sequence[str]] = None


class YOLODetector:
    """Wraps ONNX Runtime inference for YOLO-style object detectors."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        self.class_names = COCO_CLASS_NAMES
        self.class_ids_filter = self._resolve_class_filter(self.config.class_filter)

        providers = self._resolve_providers(self.config.providers)
        self.session, self.active_providers = self._create_session_with_fallback(providers)
        self.primary_provider = self.active_providers[0] if self.active_providers else 'CPUExecutionProvider'
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = self.config.input_size

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        if frame_rgb is None or frame_rgb.ndim != 3:
            raise ValueError("Expected an RGB frame with shape (H, W, 3).")

        image, ratio, pad = _letterbox(frame_rgb, self.input_size)
        tensor = image.transpose((2, 0, 1))[None].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor)

        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self._postprocess(outputs, ratio, pad, frame_rgb.shape[:2])

    # Internal helpers -----------------------------------------------------

    def _create_session_with_fallback(
        self,
        resolved_providers: Sequence[str] | None,
    ) -> Tuple[ort.InferenceSession, Tuple[str, ...]]:
        candidates = self._provider_candidates(resolved_providers)
        errors: List[str] = []
        for candidate in candidates:
            try:
                session = ort.InferenceSession(
                    str(self.config.model_path),
                    providers=candidate,
                )
                active = tuple(session.get_providers())
                return session, active
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"{tuple(candidate)}: {exc}")
        try:
            session = ort.InferenceSession(str(self.config.model_path))
            active = tuple(session.get_providers())
            return session, active
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"default: {exc}")
            detail = "; ".join(errors)
            raise RuntimeError(f"Failed to initialize ONNX Runtime session: {detail}") from exc

    def _provider_candidates(
        self,
        resolved_providers: Sequence[str] | None,
    ) -> List[List[str]]:
        candidates: List[List[str]] = []
        if self.config.providers:
            candidates.append(list(self.config.providers))
        if platform.system().lower() == "windows":
            candidates.append(["DmlExecutionProvider", "CPUExecutionProvider"])
        if resolved_providers:
            candidates.append(list(resolved_providers))
        candidates.append(["CPUExecutionProvider"])
        unique: List[List[str]] = []
        seen: set[Tuple[str, ...]] = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def describe_backend(self) -> str:
        provider = self.primary_provider
        if provider == "DmlExecutionProvider":
            return "DirectML (GPU)"
        if provider == "CUDAExecutionProvider":
            return "CUDA (GPU)"
        if provider == "ROCMExecutionProvider":
            return "ROCm (GPU)"
        if provider == "AzureExecutionProvider":
            return "Azure (cloud)"
        return provider

    def _postprocess(
        self,
        raw_output: np.ndarray,
        ratio: Tuple[float, float],
        pad: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> List[Detection]:
        predictions = _reshape_yolo_output(raw_output)
        if predictions.size == 0:
            return []

        boxes_xywh = predictions[:, :4]
        scores = predictions[:, 4:]
        if scores.size == 0:
            return []

        num_classes = len(self.class_names)
        if scores.shape[1] == num_classes + 1:
            objectness = scores[:, 0]
            class_scores = scores[:, 1:]
        elif scores.shape[1] == num_classes:
            objectness = None
            class_scores = scores
        else:
            # Fallback for models where the output layout does not match expectations exactly.
            if scores.shape[1] > num_classes:
                objectness = scores[:, 0]
                class_scores = scores[:, 1:]
            else:
                objectness = None
                class_scores = scores

        if class_scores.size == 0:
            return []

        class_ids = np.argmax(class_scores, axis=1)
        best_class_scores = class_scores[np.arange(len(class_ids)), class_ids]
        if objectness is not None:
            cls_conf = objectness * best_class_scores
        else:
            cls_conf = best_class_scores

        mask = cls_conf >= self.config.confidence_threshold
        if self.class_ids_filter is not None:
            mask &= np.isin(class_ids, list(self.class_ids_filter))
        boxes_xywh = boxes_xywh[mask]
        cls_conf = cls_conf[mask]
        class_ids = class_ids[mask]

        if boxes_xywh.size == 0:
            return []

        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = _scale_boxes(boxes_xyxy, ratio, pad, original_shape)

        keep = _nms(boxes_xyxy, cls_conf, self.config.iou_threshold)
        frame_height, frame_width = original_shape
        detections: List[Detection] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            class_id = int(class_ids[idx])
            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            if label == "car":
                label = "light vehicle"
            confidence = float(cls_conf[idx])
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area_ratio = (width * height) / max(1.0, frame_width * frame_height)
            height_ratio = height / max(frame_height, 1.0)
            # Ignore detections that are far too small to be real vehicles (e.g., pedestrians or noise).
            if area_ratio < MIN_VEHICLE_AREA_RATIO or height_ratio < MIN_VEHICLE_HEIGHT_RATIO:
                continue
            if label == "bus":
                confidence *= BUS_CONFIDENCE_SCALE
            elif label == "truck":
                aspect_ratio = width / max(height, 1.0)
                if area_ratio >= TRUCK_LARGE_MIN_AREA or aspect_ratio >= TRUCK_LARGE_MIN_ASPECT:
                    confidence = max(confidence, 0.85)
                else:
                    confidence *= TRUCK_CONFIDENCE_SCALE
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    class_id=class_id,
                )
            )
        return detections

    def _resolve_class_filter(self, class_filter: Optional[Sequence[str]]) -> Optional[Sequence[int]]:
        if not class_filter:
            return None
        name_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(self.class_names)}
        result: List[int] = []
        for label in class_filter:
            if label not in name_to_id:
                continue
            result.append(name_to_id[label])
        return result if result else None

    @staticmethod
    def _resolve_providers(explicit: Optional[Sequence[str]]) -> List[str]:
        available = set(ort.get_available_providers())
        if explicit:
            chosen = [provider for provider in explicit if provider in available]
            if chosen:
                return chosen
        preferred = [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]


# Utility functions -------------------------------------------------------

def _letterbox(
    image: np.ndarray,
    new_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    shape = image.shape[:2]
    if isinstance(new_size, int):
        new_shape = (new_size, new_size)
    else:
        new_shape = new_size

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, (ratio, ratio), (dw, dh)


def _reshape_yolo_output(raw_output: np.ndarray) -> np.ndarray:
    if raw_output.ndim == 3:
        raw_output = np.squeeze(raw_output, axis=0)
    if raw_output.ndim == 2:
        if raw_output.shape[0] <= raw_output.shape[1] and raw_output.shape[0] < 128:
            raw_output = raw_output.transpose()
        predictions = raw_output
    elif raw_output.ndim == 3:
        if raw_output.shape[1] < raw_output.shape[2]:
            predictions = raw_output.transpose(0, 2, 1).reshape(-1, raw_output.shape[1])
        else:
            predictions = raw_output.reshape(-1, raw_output.shape[-1])
    else:
        predictions = raw_output.reshape(-1, raw_output.shape[-1])
    return predictions


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.zeros_like(boxes)
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def _scale_boxes(
    boxes: np.ndarray,
    ratio: Tuple[float, float],
    pad: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, [0, 2]] /= ratio[0]
    boxes[:, [1, 3]] /= ratio[1]
    height, width = original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        idx = order[0]
        keep.append(int(idx))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[idx] + areas[order[1:]] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )

        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    return keep

_COLOR_LOW = (0, 0, 255)      # Red for confidence < 0.6
_COLOR_MED = (0, 165, 255)    # Orange for 0.6 <= confidence < 0.75
_COLOR_HIGH = (0, 255, 0)     # Green otherwise


def _color_for_confidence(confidence: float) -> Tuple[int, int, int]:
    if confidence < 0.6:
        return _COLOR_LOW
    if confidence < 0.75:
        return _COLOR_MED
    return _COLOR_HIGH


def draw_detections(frame_rgb: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    annotated = frame_rgb.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box)
        color = _color_for_confidence(float(detection.confidence))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_label = f"{detection.label} {detection.confidence:.2f}"
        baseline = max(y1 - 5, 0)
        cv2.putText(
            annotated,
            text_label,
            (x1, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


__all__ = [
    "Detection",
    "DetectorConfig",
    "YOLODetector",
    "COCO_CLASS_NAMES",
    "VEHICLE_CLASS_NAMES",
    "draw_detections",
]



# ===========================================================================
# FILE START: src\obj_tracking.py
# ===========================================================================

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





# ===========================================================================
# FILE START: src\OpenCV_desktopTest.py
# ===========================================================================

import math
import time
import ctypes
from collections import deque
from ctypes import wintypes
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mss
from ultralytics import YOLO

# -------------------- Win32 helpers --------------------
user32 = ctypes.windll.user32
GA_ROOT = 2
VK_LBUTTON = 0x01

# -------------------- Tunables --------------------
CONF_THRESHOLD = 0.55
PERSON_CLASS_ID = 0

# speed
IMGSZ = 640           # YOLO input size
FRAME_SKIP = 1        # process every (FRAME_SKIP+1)th frame; 1 => process 1, skip 1
USE_V8N = True        # use nano model for speed

# tracking / association
PIXELS_PER_METER = 50.0
MAX_MISSED_FRAMES = 30
ASSIGNMENT_MAX_DISTANCE_PX = 180.0

# appearance (hist on torso ROI in HSV)
H_BINS, S_BINS = 10, 4
APPEARANCE_WEIGHT = 180.0
APPEARANCE_SMOOTHING = 0.2
MIN_HIST_AREA = 900

# color signature (HSV mean, not BGR)
COLOR_MATCH_WEIGHT = 60.0

# collision guard
COLLISION_FEET_DISTANCE_PX = 140.0
SWAP_DEBOUNCE_FRAMES = 8           # need N consecutive frames of better pairing to allow swap during collision
SWAP_MARGIN = 40.0                 # total-cost improvement required to consider swap in collision

PLAYER_NAMES = ("Player A", "Player B")
PLAYER_COLORS = ((0, 255, 0), (0, 165, 255))

COURT_WIDTH_METERS = 6.4
COURT_LENGTH_METERS = 9.75
CALIBRATION = {
    "enabled": False,
    "far_wall": {"y": None, "left": None, "right": None},
    "near_wall": {"y": None, "left": None, "right": None},
}

# -------------------- Kalman --------------------
class KalmanFilter2D:
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-1) -> None:
        self.dt = 1.0
        self.A = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def _rebuild_transition(self, dt: float) -> None:
        self.A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    def reset(self) -> None:
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False

    def predict(self, dt: float) -> Optional[Tuple[float, float]]:
        if not self.initialized:
            return None
        if abs(dt - self.dt) > 1e-6:
            self.dt = dt
            self._rebuild_transition(dt)
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return float(self.x[0,0]), float(self.x[1,0])

    def correct(self, point: Tuple[float, float]) -> Tuple[float, float]:
        z = np.array([[point[0]],[point[1]]], dtype=np.float32)
        if not self.initialized:
            self.x[0,0] = point[0]
            self.x[1,0] = point[1]
            self.initialized = True
        else:
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y = z - (self.H @ self.x)
            self.x = self.x + (K @ y)
            I = np.eye(self.P.shape[0], dtype=np.float32)
            self.P = (I - K @ self.H) @ self.P
        return float(self.x[0,0]), float(self.x[1,0])

# -------------------- Tracking --------------------
class PlayerTrack:
    def __init__(self, name: str, color: Tuple[int, int, int]) -> None:
        self.name = name
        self.color = color
        self.measurement_px: Optional[Tuple[float, float]] = None
        self.filtered_px: Optional[Tuple[float, float]] = None
        self.filtered_m: Optional[Tuple[float, float]] = None
        self.predicted_px: Optional[Tuple[float, float]] = None
        self.appearance_hist: Optional[np.ndarray] = None
        self.color_signature: Optional[np.ndarray] = None
        self.total_distance_pixels: float = 0.0
        self.total_distance_meters_accum: float = 0.0
        self.missed_frames: int = 0
        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.last_confidence: Optional[float] = None
        self.path: Deque[Tuple[int, int]] = deque(maxlen=120)
        self.filter = KalmanFilter2D()
        # collision swap protection
        self._swap_streak = 0

    def begin_frame(self, dt: float) -> None:
        prediction = self.filter.predict(dt)
        self.predicted_px = prediction if prediction is not None else self.filtered_px

    def update(self, measurement_px, box, confidence, hist, color_signature=None) -> None:
        prev_filtered_px = self.filtered_px
        prev_filtered_m = self.filtered_m
        prediction = self.predicted_px

        filtered_px = self.filter.correct(measurement_px)

        if prev_filtered_px is not None:
            self.total_distance_pixels += math.hypot(filtered_px[0]-prev_filtered_px[0],
                                                     filtered_px[1]-prev_filtered_px[1])
        elif prediction is not None:
            self.total_distance_pixels += math.hypot(filtered_px[0]-prediction[0],
                                                     filtered_px[1]-prediction[1])

        self.measurement_px = measurement_px
        self.filtered_px = filtered_px
        self.filtered_m = pixel_to_court_coords(*filtered_px)
        self.last_box = box
        self.last_confidence = confidence
        self.path.append((int(filtered_px[0]), int(filtered_px[1])))
        self.missed_frames = 0
        self.predicted_px = filtered_px
        self._update_appearance(hist)
        self._update_color(color_signature)
        self._swap_streak = 0  # successful update resets streak

        if self.filtered_m is not None:
            if prev_filtered_m is not None:
                self.total_distance_meters_accum += math.hypot(self.filtered_m[0]-prev_filtered_m[0],
                                                               self.filtered_m[1]-prev_filtered_m[1])
            elif prediction is not None:
                prev_m = pixel_to_court_coords(*prediction)
                if prev_m is not None:
                    self.total_distance_meters_accum += math.hypot(self.filtered_m[0]-prev_m[0],
                                                                   self.filtered_m[1]-prev_m[1])

    def _update_appearance(self, hist: Optional[np.ndarray]) -> None:
        if hist is None or hist.size == 0:
            return
        hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        if self.appearance_hist is None:
            self.appearance_hist = hist
        else:
            blended = (1.0 - APPEARANCE_SMOOTHING) * self.appearance_hist + APPEARANCE_SMOOTHING * hist
            self.appearance_hist = cv2.normalize(blended, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    def _update_color(self, color_signature: Optional[np.ndarray]) -> None:
        if color_signature is None:
            return
        if self.color_signature is None:
            self.color_signature = color_signature
        else:
            self.color_signature = (1.0 - APPEARANCE_SMOOTHING) * self.color_signature + APPEARANCE_SMOOTHING * color_signature

    def mark_missed(self) -> None:
        self.missed_frames += 1
        if self.predicted_px is not None:
            self.filtered_px = self.predicted_px
            self.filtered_m = pixel_to_court_coords(*self.predicted_px)
        if self.missed_frames > MAX_MISSED_FRAMES:
            self.reset()

    def bump_swap_streak(self):
        self._swap_streak += 1

    def reset_swap_streak(self):
        self._swap_streak = 0

    @property
    def swap_streak(self) -> int:
        return self._swap_streak

    def reset(self) -> None:
        self.measurement_px = None
        self.filtered_px = None
        self.filtered_m = None
        self.predicted_px = None
        self.appearance_hist = None
        self.color_signature = None
        self.total_distance_pixels = 0.0
        self.total_distance_meters_accum = 0.0
        self.missed_frames = 0
        self.last_box = None
        self.last_confidence = None
        self.path.clear()
        self.filter.reset()
        self._swap_streak = 0

    @property
    def total_distance_meters(self) -> float:
        if CALIBRATION["enabled"] and self.total_distance_meters_accum > 0.0:
            return self.total_distance_meters_accum
        return 0.0 if PIXELS_PER_METER <= 0 else self.total_distance_pixels / PIXELS_PER_METER

players: List[PlayerTrack] = [PlayerTrack(n, c) for n, c in zip(PLAYER_NAMES, PLAYER_COLORS)]

# -------------------- Utils --------------------
def try_set_process_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass

def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()

def get_window_rect(hwnd: int) -> Optional[Dict[str, int]]:
    rect = wintypes.RECT()
    if not user32.IsWindow(hwnd):
        return None
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    return {"top": rect.top, "left": rect.left, "width": width, "height": height}

def wait_for_window_selection() -> int:
    try_set_process_dpi_aware()
    print("Click on the window you want to monitor, then release the mouse button...")
    lb_was_down = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
    while True:
        time.sleep(0.01)
        lb_down = bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)
        if lb_down and not lb_was_down:
            cursor = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(cursor))
            hwnd = user32.WindowFromPoint(cursor)
            if not hwnd:
                print("No window detected. Click again.")
            else:
                hwnd = user32.GetAncestor(hwnd, GA_ROOT)
                rect = get_window_rect(hwnd)
                if rect:
                    title = get_window_title(hwnd) or "Untitled window"
                    print(f"Selected window: {title} ({rect['width']}x{rect['height']})")
                    return hwnd
                print("Could not read window bounds. Try another window.")
        lb_was_down = lb_down

def calibration_ready() -> bool:
    far = CALIBRATION["far_wall"]; near = CALIBRATION["near_wall"]
    required = (far["y"], far["left"], far["right"], near["y"], near["left"], near["right"])
    return all(value is not None for value in required)

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def depth_alpha(y: float) -> float:
    far_y = CALIBRATION["far_wall"]["y"]; near_y = CALIBRATION["near_wall"]["y"]
    if far_y is None or near_y is None or near_y == far_y:
        return 0.0
    t = (y - far_y) / (near_y - far_y)
    return clamp(t, 0.0, 1.0)

def pixel_to_court_coords(x: float, y: float) -> Optional[Tuple[float, float]]:
    if not CALIBRATION["enabled"]:
        return None
    if not calibration_ready():
        raise ValueError("Calibration enabled but pixel references incomplete.")
    far = CALIBRATION["far_wall"]; near = CALIBRATION["near_wall"]
    alpha = depth_alpha(y)
    left_boundary = lerp(far["left"], near["left"], alpha)
    right_boundary = lerp(far["right"], near["right"], alpha)
    width_pixels = right_boundary - left_boundary
    if width_pixels <= 0:
        return None
    x_meters = (x - left_boundary) * (COURT_WIDTH_METERS / width_pixels)
    y_meters = alpha * COURT_LENGTH_METERS
    return x_meters, y_meters

def compute_feet_point(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)

def clip_box(box, width, height) -> Optional[Tuple[int, int, int, int]]:
    x1,y1,x2,y2 = box
    x1 = max(0, min(x1, width - 1)); x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1)); y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1,y1,x2,y2

def _torso_roi(box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    """Upper-middle region of the box (shirt area), reduces background."""
    x1,y1,x2,y2 = box
    h = y2 - y1; w = x2 - x1
    top = y1 + int(0.15*h)
    bottom = y1 + int(0.55*h)
    left = x1 + int(0.20*w)
    right = x1 + int(0.80*w)
    return left, top, right, bottom

def compute_histogram_torso_hs(frame: np.ndarray, box: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if box is None: return None
    rx1, ry1, rx2, ry2 = _torso_roi(box)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0 or (roi.shape[0] * roi.shape[1]) < MIN_HIST_AREA:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [H_BINS, S_BINS], [0,180, 0,256])
    hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)

def compute_color_signature_hsv(frame: np.ndarray, box: Optional[Tuple[int,int,int,int]]) -> Optional[np.ndarray]:
    if box is None: return None
    rx1, ry1, rx2, ry2 = _torso_roi(box)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = cv2.mean(hsv)[:3]
    return np.array(mean, dtype=np.float32)

def _hist_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.5  # neutral distance
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))

def association_cost(track, detection) -> float:
    expected = track.predicted_px or track.filtered_px
    if expected is None:
        return float("inf")
    pos = detection["feet_px"]
    dist = math.hypot(pos[0]-expected[0], pos[1]-expected[1])
    if dist > ASSIGNMENT_MAX_DISTANCE_PX:
        return float("inf")
    cost = dist
    # appearance distances
    cost += APPEARANCE_WEIGHT * _hist_dist(track.appearance_hist, detection["hist"])
    if track.color_signature is not None and detection.get("color_signature") is not None:
        color_dist = np.linalg.norm(track.color_signature - detection["color_signature"]) / 255.0
        cost += COLOR_MATCH_WEIGHT * color_dist
    return cost

def _feet_distance_px(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def associate_detections(tracks: List[PlayerTrack], detections: List[Dict[str, object]]) -> None:
    # mark missed if no detections
    if not detections:
        for t in tracks: t.mark_missed()
        return

    remaining = detections.copy()
    assigned: List[PlayerTrack] = []

    # Bootstrap with left/right if uninitialized
    uninitialized = [t for t in tracks if t.filtered_px is None and t.predicted_px is None]
    if uninitialized:
        remaining.sort(key=lambda d: d["feet_px"][0])
        for t, det in zip(uninitialized, remaining):
            t.update(det["feet_px"], det["box"], det["confidence"], det["hist"], det.get("color_signature"))
            assigned.append(t)
        remaining = remaining[len(uninitialized):]

    active = [t for t in tracks if t not in assigned]
    if active and remaining:
        remaining.sort(key=lambda d: d["confidence"], reverse=True)
        candidate = remaining[:len(active)]

        if len(active) == 1:
            t = active[0]; d = candidate[0]
            c = association_cost(t, d)
            if math.isfinite(c):
                t.update(d["feet_px"], d["box"], d["confidence"], d["hist"], d.get("color_signature"))
                assigned.append(t); remaining.remove(d)
        elif len(active) >= 2 and len(candidate) >= 2:
            ta, tb = active[:2]; da, db = candidate[:2]
            # compute both pairings
            c_aa = association_cost(ta, da); c_bb = association_cost(tb, db)
            c_ab = association_cost(ta, db); c_ba = association_cost(tb, da)

            best_pairs = None
            best_cost = float("inf")

            # default: keep previous ID mapping
            keep_pairs = [(ta, da), (tb, db)]
            keep_total = (c_aa if math.isfinite(c_aa) else 1e9) + (c_bb if math.isfinite(c_bb) else 1e9)

            swap_pairs = [(ta, db), (tb, da)]
            swap_total = (c_ab if math.isfinite(c_ab) else 1e9) + (c_ba if math.isfinite(c_ba) else 1e9)

            # Collision guard: if feet of detections are very close -> prefer KEEP unless SWAP is much better for several frames
            feet_dist = _feet_distance_px(da["feet_px"], db["feet_px"])
            collision = feet_dist < COLLISION_FEET_DISTANCE_PX

            if collision:
                # If swap is clearly better, increment streak; otherwise reset
                if swap_total + SWAP_MARGIN < keep_total:
                    # bump streak on both tracks (they share the event)
                    ta.bump_swap_streak(); tb.bump_swap_streak()
                    if min(ta.swap_streak, tb.swap_streak) >= SWAP_DEBOUNCE_FRAMES:
                        best_pairs, best_cost = swap_pairs, swap_total
                    else:
                        best_pairs, best_cost = keep_pairs, keep_total
                else:
                    ta.reset_swap_streak(); tb.reset_swap_streak()
                    best_pairs, best_cost = keep_pairs, keep_total
            else:
                # No collision → pick lower cost and reset streaks
                ta.reset_swap_streak(); tb.reset_swap_streak()
                if swap_total < keep_total:
                    best_pairs, best_cost = swap_pairs, swap_total
                else:
                    best_pairs, best_cost = keep_pairs, keep_total

            for t, d in best_pairs:
                t.update(d["feet_px"], d["box"], d["confidence"], d["hist"], d.get("color_signature"))
                assigned.append(t)
                if d in remaining: remaining.remove(d)

        # any leftover tracks
        leftover = [t for t in active if t not in assigned]
        for t in leftover:
            best_d, best_c = None, float("inf")
            for d in remaining:
                c = association_cost(t, d)
                if math.isfinite(c) and c < best_c:
                    best_c, best_d = c, d
            if best_d is not None:
                t.update(best_d["feet_px"], best_d["box"], best_d["confidence"], best_d["hist"], best_d.get("color_signature"))
                assigned.append(t); remaining.remove(best_d)

    for t in tracks:
        if t not in assigned:
            t.mark_missed()

def reset_players_state() -> None:
    for t in players: t.reset()

# -------------------- Main --------------------
def main() -> None:
    reset_players_state()
    model = YOLO("yolov8n.pt" if USE_V8N else "yolov8s.pt")
    sct = mss.mss()
    hwnd = wait_for_window_selection()

    cv2.namedWindow("YOLO Desktop Detection", cv2.WINDOW_NORMAL)
    last_time = time.time()
    frame_idx = 0

    while True:
        now = time.time()
        dt = max(now - last_time, 1e-3)
        last_time = now

        region = get_window_rect(hwnd)
        if region is None:
            print("Selected window is no longer available. Exiting.")
            break

        for t in players:
            t.begin_frame(dt)

        img = np.array(sct.grab(region))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # Frame skip for speed
        run_yolo = (frame_idx % (FRAME_SKIP + 1) == 0)
        detections: List[Dict[str, object]] = []

        if run_yolo:
            # NOTE: Pass BGR numpy array directly; restrict to persons and target input size
            results = model(frame_bgr, classes=[PERSON_CLASS_ID], imgsz=IMGSZ, verbose=False)
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, classes):
                    if conf < CONF_THRESHOLD or cls != PERSON_CLASS_ID:
                        continue
                    box = clip_box((x1,y1,x2,y2), w, h)
                    if box is None: continue
                    feet = compute_feet_point(box)

                    # torso HSV hist + HSV mean as color signature
                    hist = compute_histogram_torso_hs(frame_bgr, box)
                    color_sig = compute_color_signature_hsv(frame_bgr, box)

                    detections.append({
                        "box": box,
                        "confidence": float(conf),
                        "feet_px": feet,
                        "hist": hist,
                        "color_signature": color_sig,
                    })

            associate_detections(players, detections)

        # draw overlays
        for t in players:
            if t.last_box is None:
                continue
            x1,y1,x2,y2 = t.last_box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), t.color, 2)
            dist_m = t.total_distance_meters
            conf = t.last_confidence if t.last_confidence is not None else 0.0
            text_y = min(y2 + 20, h - 20)
            label = f"{t.name} {conf:.2f} ({dist_m:.1f} m)"
            cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
            for i in range(1, len(t.path)):
                cv2.line(annotated, t.path[i-1], t.path[i], t.color, 2)

        y = 60
        for t in players:
            summary = f"{t.name}: {t.total_distance_meters:.1f} m"
            cv2.putText(annotated, summary, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, t.color, 2)
            y += 25

        cv2.imshow("YOLO Desktop Detection", annotated)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("r"), ord("R")):
            reset_players_state()
            last_time = time.time()
            frame_idx = 0
            continue
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# ===========================================================================
# FILE START: src\tracking.py
# ===========================================================================

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



# ===========================================================================
# FILE START: src\vehicle_dataset.py
# ===========================================================================

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import time


@dataclass(slots=True)
class VehicleRecord:
    timestamp: float  # seconds since start of video
    object_id: int
    classification: str
    confidence: float
    source: str  # 'yolo' | 'vlm' | 'chatgpt'
    line_label: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class VehicleDataset:
    """Thread-safe in-memory dataset of all vehicle detections and classifications."""

    def __init__(self) -> None:
        self._records: Dict[int, VehicleRecord] = {}
        self._lock = threading.Lock()

    def add_or_update(self, record: VehicleRecord) -> None:
        with self._lock:
            existing = self._records.get(record.object_id)
            if existing is None:
                self._records[record.object_id] = record
                return
            if record.source != "yolo":
                self._records[record.object_id] = record
                return

            should_replace = False
            if record.confidence > existing.confidence:
                should_replace = True
            if record.classification != existing.classification:
                should_replace = True

            incoming_label = record.line_label or existing.line_label
            if existing.line_label != incoming_label:
                record.line_label = incoming_label
                should_replace = True

            if should_replace:
                self._records[record.object_id] = record

    def get_low_conf(self, threshold: float = 0.5) -> List[VehicleRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.confidence < threshold]

    def all_records(self) -> List[VehicleRecord]:
        with self._lock:
            return list(self._records.values())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self, interval_s: int = 900) -> Dict[str, Dict[str, int]]:
        """Return counts by class for 15-min (900s) intervals."""
        with self._lock:
            if not self._records:
                return {}

            buckets: Dict[str, Dict[str, int]] = {}
            for r in self._records.values():
                bucket_index = int(r.timestamp // interval_s)
                bucket_label = f"{bucket_index * 15:02d}-{(bucket_index + 1) * 15:02d}min"
                class_counts = buckets.setdefault(bucket_label, {})
                class_counts[r.classification] = class_counts.get(r.classification, 0) + 1
            return buckets

    def detected_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r.line_label)

    def identified_count(self, threshold: float = 0.5) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r.line_label and r.confidence >= threshold)


# Example standalone test
if __name__ == "__main__":
    ds = VehicleDataset()
    start = time.time()
    ds.add_or_update(VehicleRecord(timestamp=10.0, object_id=1, classification="car", confidence=0.4, source="yolo"))
    ds.add_or_update(VehicleRecord(timestamp=12.0, object_id=2, classification="truck", confidence=0.8, source="yolo"))
    ds.add_or_update(VehicleRecord(timestamp=15.0, object_id=1, classification="car", confidence=0.7, source="vlm"))

    print("All Records:", [r.as_dict() for r in ds.all_records()])
    print("Low Confidence:", [r.as_dict() for r in ds.get_low_conf(0.6)])
    print("Summary:", ds.summary())
    print(f"Detected: {ds.detected_count()} | Identified: {ds.identified_count()}")

# Global shared dataset instance
DATASET = VehicleDataset()



# ===========================================================================
# FILE START: src\video_processing.py
# ===========================================================================

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



# ===========================================================================
# FILE START: src\video_worker.py
# ===========================================================================

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



# ===========================================================================
# FILE START: src\vlm_queue.py
# ===========================================================================

"""Detection staging queues and simulated VLM enrichment worker."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from vehicle_dataset import DATASET, VehicleRecord

stage_one_queue: "queue.Queue[VehicleRecord]" = queue.Queue()
low_conf_queue: "queue.Queue[VehicleRecord]" = queue.Queue()
stop_flag = threading.Event()


def enqueue_stage_one(record: VehicleRecord) -> None:
    """Push a freshly detected vehicle record onto the stage-one queue."""
    stage_one_queue.put(record)


def dequeue_stage_one(timeout: Optional[float] = None) -> VehicleRecord:
    """Convenience helper for tests or future pipeline steps."""
    return stage_one_queue.get(timeout=timeout)


def _vlm_worker() -> None:
    """Poll the low-confidence queue and simulate a VLM enrichment pass."""
    while not stop_flag.is_set():
        try:
            record = low_conf_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            time.sleep(0.3)
            record.confidence = min(1.0, record.confidence + 0.25)
            record.source = "vlm"
            DATASET.add_or_update(record)
            line_info = record.line_label or "-"
            print(
                f"[VLM] track {record.object_id} ({record.classification}) "
                f"conf={record.confidence:.2f} line={line_info} t={record.timestamp:.2f}s"
            )
        finally:
            low_conf_queue.task_done()


def enqueue_low_confidence(record: VehicleRecord) -> None:
    """Queue a low-confidence record for VLM enrichment."""
    line_info = record.line_label or "-"
    print(
        f"[Stage1->VLM] track {record.object_id} ({record.classification}) "
        f"conf={record.confidence:.2f} line={line_info} t={record.timestamp:.2f}s"
    )
    low_conf_queue.put(record)


_thread = threading.Thread(target=_vlm_worker, daemon=True)
_thread.start()


def stop_vlm_worker() -> None:
    stop_flag.set()


