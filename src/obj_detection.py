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
