from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from io import BytesIO


@dataclass(frozen=True)
class ImagePreparationPolicy:
    enabled: bool = True
    max_long_edge: int = 3072
    max_pixels: int = 4_014_080
    max_bytes: int = 4_000_000
    jpeg_quality: int = 92
    min_jpeg_quality: int = 72

    def __post_init__(self) -> None:
        if self.max_long_edge < 256:
            raise ValueError("max_long_edge must be at least 256")
        if self.max_pixels < 65_536:
            raise ValueError("max_pixels must be at least 65536")
        if self.max_bytes < 32_768:
            raise ValueError("max_bytes must be at least 32768")
        if not 30 <= self.min_jpeg_quality <= self.jpeg_quality <= 95:
            raise ValueError("JPEG quality must satisfy 30 <= min <= quality <= 95")


def prepare_image_for_vlm(
    image_bytes: bytes,
    *,
    policy: ImagePreparationPolicy | None = None,
) -> tuple[bytes, dict]:
    """Create a bounded inference copy while preserving the original evidence bytes."""
    policy = policy or ImagePreparationPolicy()
    source_sha = hashlib.sha256(image_bytes).hexdigest()
    metadata = {
        "policy": asdict(policy),
        "source_bytes": len(image_bytes),
        "source_sha256": source_sha,
        "processed": False,
    }
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            source_width, source_height = source.size
            source_format = (source.format or "unknown").lower()
            metadata.update({
                "source_width": source_width,
                "source_height": source_height,
                "source_format": source_format,
            })
            if not policy.enabled:
                metadata.update({
                    "output_bytes": len(image_bytes),
                    "output_width": source_width,
                    "output_height": source_height,
                    "output_format": source_format,
                    "output_sha256": source_sha,
                    "scale": 1.0,
                    "reason": "disabled",
                })
                return image_bytes, metadata
            needs_resize = (
                max(source_width, source_height) > policy.max_long_edge
                or source_width * source_height > policy.max_pixels
            )
            needs_compression = len(image_bytes) > policy.max_bytes
            if not needs_resize and not needs_compression:
                metadata.update({
                    "output_bytes": len(image_bytes),
                    "output_width": source_width,
                    "output_height": source_height,
                    "output_format": source_format,
                    "output_sha256": source_sha,
                    "scale": 1.0,
                    "reason": "within_budget",
                })
                return image_bytes, metadata

            scale = min(
                1.0,
                policy.max_long_edge / max(source_width, source_height),
                math.sqrt(policy.max_pixels / (source_width * source_height)),
            )
            image = source.convert("RGB")
            if scale < 1:
                image = image.resize(
                    (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
                    Image.Resampling.LANCZOS,
                )

            output = b""
            used_quality = policy.jpeg_quality
            while True:
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=used_quality,
                    optimize=True,
                    progressive=True,
                )
                output = buffer.getvalue()
                if len(output) <= policy.max_bytes or used_quality <= policy.min_jpeg_quality:
                    break
                used_quality = max(policy.min_jpeg_quality, used_quality - 7)

            # Extremely noisy scans can remain large even at minimum quality.
            # Reduce both axes uniformly, preserving normalized bbox coordinates.
            while len(output) > policy.max_bytes and min(image.size) > 256:
                reduction = max(0.65, math.sqrt(policy.max_bytes / len(output)) * 0.95)
                next_size = (
                    max(256, round(image.width * reduction)),
                    max(256, round(image.height * reduction)),
                )
                if next_size == image.size:
                    break
                image = image.resize(next_size, Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=policy.min_jpeg_quality,
                    optimize=True,
                    progressive=True,
                )
                output = buffer.getvalue()

            metadata.update({
                "processed": True,
                "output_bytes": len(output),
                "output_width": image.width,
                "output_height": image.height,
                "output_format": "jpeg",
                "output_sha256": hashlib.sha256(output).hexdigest(),
                "scale": round(image.width / source_width, 6),
                "jpeg_quality": used_quality,
                "reason": "resize" if needs_resize else "byte_budget",
                "normalized_bbox_preserved": True,
            })
            return output, metadata
    except Exception as exc:
        metadata.update({
            "output_bytes": len(image_bytes),
            "output_width": None,
            "output_height": None,
            "output_format": "unknown",
            "output_sha256": source_sha,
            "scale": 1.0,
            "reason": "decode_fallback",
            "decode_error": str(exc),
        })
        return image_bytes, metadata
