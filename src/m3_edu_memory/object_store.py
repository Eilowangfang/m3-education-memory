from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .media import image_mime_type


EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class StoredObject:
    sha256: str
    path: Path
    uri: str
    mime_type: str
    byte_size: int


class LocalObjectStore:
    """Content-addressed immutable storage with atomic local writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def put_image(self, data: bytes) -> StoredObject:
        mime = image_mime_type(data)
        if mime not in EXTENSIONS:
            raise ValueError(f"Unsupported image type: {mime}")
        digest = hashlib.sha256(data).hexdigest()
        destination = self.root / "sha256" / digest[:2] / f"{digest}{EXTENSIONS[mime]}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            handle, temporary = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=destination.parent
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return StoredObject(
            sha256=digest,
            path=destination,
            uri=destination.as_uri(),
            mime_type=mime,
            byte_size=len(data),
        )
