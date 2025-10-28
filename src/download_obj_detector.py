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


