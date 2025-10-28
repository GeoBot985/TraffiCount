"""File handling helpers for TraffiCount."""
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
