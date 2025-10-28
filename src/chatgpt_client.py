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
