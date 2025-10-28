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
