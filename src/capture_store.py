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
