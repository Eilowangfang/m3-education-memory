from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .image_preprocessing import ImagePreparationPolicy, prepare_image_for_vlm


def load_embedding_profiles(path: str | Path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Embedding config must contain a non-empty profiles object")
    for name, profile in profiles.items():
        if "api_key" in profile:
            raise ValueError(
                f"Embedding profile {name!r} contains api_key; use api_key_env"
            )
        for field in ("endpoint", "model", "api_key_env"):
            if not isinstance(profile.get(field), str) or not profile[field].strip():
                raise ValueError(f"Embedding profile {name!r} requires {field}")
        if int(profile.get("dimension", 0)) <= 0:
            raise ValueError(f"Embedding profile {name!r} requires dimension")
        if profile.get("input_mode", "text") not in {"text", "text_image"}:
            raise ValueError(
                f"Embedding profile {name!r} input_mode must be text or text_image"
            )
        image_options = profile.get("image_preprocessing") or {}
        if not isinstance(image_options, dict):
            raise ValueError(
                f"Embedding profile {name!r} image_preprocessing must be an object"
            )
        ImagePreparationPolicy(**image_options)
    return profiles


@dataclass
class ArkMultimodalEmbeddingClient:
    endpoint: str
    model: str
    api_key_env: str = "ARK_API_KEY"
    dimension: int = 1024
    timeout: int = 120
    max_retries: int = 3
    document_instructions: str = ""
    query_instructions: str = ""
    input_mode: str = "text"
    image_policy: ImagePreparationPolicy = ImagePreparationPolicy(
        max_long_edge=1600,
        max_pixels=1_600_000,
        max_bytes=1_000_000,
        jpeg_quality=88,
        min_jpeg_quality=72,
    )

    @property
    def index_fingerprint(self) -> str:
        policy = self.image_policy
        return (
            f"{self.model}:{self.input_mode}:"
            f"{policy.max_long_edge}:{policy.max_pixels}:{policy.max_bytes}:"
            f"{policy.jpeg_quality}:{policy.min_jpeg_quality}"
        )

    def _embed_input(self, input_items: list[dict], *, instructions: str) -> list[float]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env} is not configured"
            )
        payload = {
            "model": self.model,
            "encoding_format": "float",
            "dimensions": self.dimension,
            "input": input_items,
            "instructions": instructions,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        body = None
        for retry in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code not in (429, 500, 502, 503, 504) or retry + 1 >= self.max_retries:
                    raise RuntimeError(
                        f"Embedding HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if retry + 1 >= self.max_retries:
                    raise RuntimeError(f"Embedding request failed: {exc}") from exc
            time.sleep(2 ** retry)
        if body is None:
            raise RuntimeError("Embedding request failed without a response")
        data = body.get("data")
        vector = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise ValueError(
                f"Embedding response dimension mismatch: expected {self.dimension}"
            )
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        return values if norm == 0 else [value / norm for value in values]

    def _embed(self, text: str, *, instructions: str) -> list[float]:
        return self._embed_input(
            [{"type": "text", "text": text}], instructions=instructions
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_document(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, instructions=self.document_instructions)

    def embed_multimodal(
        self,
        text: str,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> list[float]:
        if self.input_mode != "text_image":
            return self.embed_document(text)
        prepared, metadata = prepare_image_for_vlm(
            image_bytes, policy=self.image_policy
        )
        output_format = str(metadata.get("output_format") or "").lower()
        prepared_mime = (
            "image/jpeg" if output_format in {"jpeg", "jpg"} else mime_type
        )
        data_url = (
            f"data:{prepared_mime};base64,"
            f"{base64.b64encode(prepared).decode('ascii')}"
        )
        return self._embed_input(
            [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": text},
            ],
            instructions=self.document_instructions,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, instructions=self.query_instructions)


def create_embedding_client(
    config_path: str | Path,
    *,
    profile_name: str,
) -> ArkMultimodalEmbeddingClient:
    profiles = load_embedding_profiles(config_path)
    if profile_name not in profiles:
        choices = ", ".join(sorted(profiles))
        raise KeyError(
            f"Unknown embedding profile {profile_name!r}; available: {choices}"
        )
    profile = profiles[profile_name]
    image_options = profile.get("image_preprocessing") or {}
    return ArkMultimodalEmbeddingClient(
        endpoint=profile["endpoint"],
        model=profile["model"],
        api_key_env=profile["api_key_env"],
        dimension=int(profile["dimension"]),
        timeout=int(profile.get("timeout", 120)),
        max_retries=int(profile.get("max_retries", 3)),
        document_instructions=str(profile.get("document_instructions", "")),
        query_instructions=str(profile.get("query_instructions", "")),
        input_mode=str(profile.get("input_mode", "text")),
        image_policy=ImagePreparationPolicy(**image_options),
    )
