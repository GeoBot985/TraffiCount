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
