from __future__ import annotations

import base64
import http.client
import json
import os
import re
import urllib.error
import urllib.request
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .media import image_mime_type
from .image_preprocessing import ImagePreparationPolicy
from .taxonomy import ERROR_TYPES


PROMPT_VERSION = "math-diagnosis-v3"


class VLMRequestError(RuntimeError):
    def __init__(self, message: str, *, request_attempts: int):
        super().__init__(message)
        self.request_attempts = request_attempts

SYSTEM_PROMPT = """You analyze a single image of a handwritten mathematics problem and solution.
Return only one JSON object matching the requested schema. Transcribe what is visibly written before judging it.
Do not invent hidden student intentions. Identify the first step that makes the reasoning invalid, if any.
Use null when the image does not support a conclusion. Bounding boxes use normalized [x1,y1,x2,y2] values from 0 to 1.
Write error explanations and correction explanations in Simplified Chinese. Keep mathematical notation exact.
When possible, include one machine-verifiable ASCII expression for the correction. Use a numeric or symbolic
equality such as (x+1)^2=x^2+2*x+1, diff(x^2,x)=2*x, or integrate(2*x,x)=x^2.
Supported functions are sin, cos, tan, exp, log, sqrt and abs. Otherwise return null.
Allowed error_type values: conceptual, assumption, algebraic_manipulation, arithmetic, notation, transcription,
omitted_step, presentation, no_actual_error, uncertain."""


def build_user_prompt(question_text: str) -> str:
    return f"""Known printed question text (may help disambiguate handwriting):
{question_text}

Analyze only the supplied image and the question above. Do not use a reference answer.
Return this JSON shape:
{{
  "question_transcription": "string",
  "solution_transcription": "string",
  "steps": [
    {{
      "step_index": 0,
      "transcription": "string",
      "normalized_latex": "string or null",
      "bbox": [0.0, 0.0, 1.0, 1.0] or null,
      "is_error": true or false or null,
      "confidence": 0.0
    }}
  ],
  "knowledge_points": ["string"],
  "has_error": true or false or null,
  "error_type": "one allowed value",
  "first_error_step": 0 or null,
  "error_explanation": "string",
  "correction": {{
    "corrected_solution": "plain-text corrected derivation suitable for drawing on an image",
    "corrected_steps": [
      {{"step_index": 0, "latex": "string", "explanation": "string"}}
    ],
    "final_answer": "string or null",
    "verification_expression": "ASCII equality, diff/integrate/limit check, or null",
    "confidence": 0.0
  }} or null,
  "confidence": 0.0,
  "requires_review": true or false
}}"""


class VisionClient(Protocol):
    provider: str
    model: str

    def analyze(
        self, *, image_bytes: bytes, question_text: str
    ) -> tuple[str, dict] | tuple[str, dict, dict]: ...


def parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM response did not contain a JSON object")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response JSON must be an object")
    return value


def validate_diagnosis(value: dict) -> dict:
    required = (
        "question_transcription", "solution_transcription", "steps",
        "knowledge_points", "has_error", "error_type", "first_error_step",
        "error_explanation", "correction", "confidence", "requires_review",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"VLM diagnosis missing fields: {', '.join(missing)}")
    if value["error_type"] not in ERROR_TYPES:
        raise ValueError(f"Unsupported error_type: {value['error_type']!r}")
    if value["has_error"] not in (True, False, None):
        raise ValueError("has_error must be true, false, or null")
    for field_name in (
        "question_transcription", "solution_transcription", "error_explanation"
    ):
        if not isinstance(value[field_name], str):
            value[field_name] = ""
            value["requires_review"] = True
    if value["has_error"] is True and not value["error_explanation"].strip():
        value["error_explanation"] = "模型未提供可用的错误说明。"
        value["requires_review"] = True
    if not isinstance(value["steps"], list):
        raise ValueError("steps must be an array")
    if not isinstance(value["knowledge_points"], list):
        raise ValueError("knowledge_points must be an array")
    correction = value["correction"]
    if value["has_error"] is True and not isinstance(correction, dict):
        raise ValueError("correction must be an object when has_error is true")
    if correction is not None:
        correction_required = (
            "corrected_solution", "corrected_steps", "final_answer",
            "verification_expression", "confidence",
        )
        correction_missing = [key for key in correction_required if key not in correction]
        if correction_missing:
            raise ValueError(
                "correction missing fields: " + ", ".join(correction_missing)
            )
        if not isinstance(correction["corrected_steps"], list):
            raise ValueError("correction.corrected_steps must be an array")
        correction_confidence = float(correction["confidence"])
        if not 0 <= correction_confidence <= 1:
            raise ValueError("correction.confidence must be between 0 and 1")
        for index, step in enumerate(correction["corrected_steps"]):
            if not isinstance(step, dict):
                raise ValueError(f"correction.corrected_steps[{index}] must be an object")
            for field in ("step_index", "latex", "explanation"):
                if field not in step:
                    raise ValueError(
                        f"correction.corrected_steps[{index}] missing {field}"
                    )
    confidence = float(value["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    for index, step in enumerate(value["steps"]):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{index}] must be an object")
        step.setdefault("step_index", index)
        step.setdefault("normalized_latex", None)
        step.setdefault("bbox", None)
        step.setdefault("is_error", None)
        step.setdefault("confidence", 0.0)
        step.setdefault("transcription", "")
        bbox = step["bbox"]
        if bbox is not None:
            try:
                normalized_bbox = [float(item) for item in bbox]
                valid_bbox = (
                    isinstance(bbox, list)
                    and len(normalized_bbox) == 4
                    and all(0 <= item <= 1 for item in normalized_bbox)
                    and normalized_bbox[0] <= normalized_bbox[2]
                    and normalized_bbox[1] <= normalized_bbox[3]
                )
            except (TypeError, ValueError):
                valid_bbox = False
                normalized_bbox = []
            if valid_bbox:
                step["bbox"] = normalized_bbox
            else:
                # A malformed or pixel-space box must not discard an otherwise
                # useful diagnosis. Omit the location and send it to review.
                step["bbox"] = None
                value["requires_review"] = True
    return value


@dataclass
class OpenAICompatibleVisionClient:
    base_url: str
    model: str
    api_key_env: str = "VLM_API_KEY"
    timeout: int = 180
    use_json_mode: bool = True
    max_retries: int = 3
    provider: str = "openai-compatible"
    image_policy: ImagePreparationPolicy = field(default_factory=ImagePreparationPolicy)
    image_detail: str = "high"
    request_options: dict[str, Any] = field(default_factory=dict)

    def analyze(self, *, image_bytes: bytes, question_text: str) -> tuple[str, dict, dict]:
        if self.image_detail not in {"low", "high", "xhigh"}:
            raise ValueError("image_detail must be low, high, or xhigh")
        api_key = None if self.api_key_env == "-" else os.environ.get(self.api_key_env)
        if self.api_key_env != "-" and not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not configured"
            )
        mime = image_mime_type(image_bytes)
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(question_text)},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": self.image_detail}},
                    ],
                },
            ],
        }
        allowed_options = {
            "thinking", "reasoning_effort", "max_tokens", "service_tier"
        }
        unsupported = set(self.request_options) - allowed_options
        if unsupported:
            raise ValueError(
                "Unsupported VLM request options: " + ", ".join(sorted(unsupported))
            )
        payload.update(self.request_options)
        if self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = None
        response_metadata: dict[str, Any] = {}
        for retry in range(self.max_retries):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    response_metadata = {
                        "http_status": getattr(response, "status", 200),
                        "request_id": (
                            response.headers.get("x-request-id")
                            or response.headers.get("x-dashscope-request-id")
                        ),
                        "request_attempts": retry + 1,
                    }
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code not in (429, 500, 502, 503, 504) or retry + 1 >= self.max_retries:
                    raise VLMRequestError(
                        f"VLM HTTP {exc.code}: {detail}", request_attempts=retry + 1
                    ) from exc
            except urllib.error.URLError as exc:
                if retry + 1 >= self.max_retries:
                    raise VLMRequestError(
                        f"VLM connection failed: {exc.reason}",
                        request_attempts=retry + 1,
                    ) from exc
            except (ConnectionError, http.client.HTTPException) as exc:
                if retry + 1 >= self.max_retries:
                    raise VLMRequestError(
                        f"VLM connection interrupted: {exc}",
                        request_attempts=retry + 1,
                    ) from exc
            except TimeoutError as exc:
                if retry + 1 >= self.max_retries:
                    raise VLMRequestError(
                        f"VLM request timed out after {self.timeout} seconds",
                        request_attempts=retry + 1,
                    ) from exc
            time.sleep(2 ** retry)
        if body is None:
            raise VLMRequestError(
                "VLM request failed without a response",
                request_attempts=self.max_retries,
            )
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        response_metadata["usage"] = body.get("usage") or {}
        return content, validate_diagnosis(parse_json_object(content)), response_metadata


@dataclass
class FixtureVisionClient:
    fixture: dict
    provider: str = "fixture"
    model: str = "fixture-math-vlm"

    def analyze(self, *, image_bytes: bytes, question_text: str) -> tuple[str, dict]:
        del image_bytes, question_text
        raw = json.dumps(self.fixture, ensure_ascii=False)
        return raw, validate_diagnosis(json.loads(raw))
