from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read FERMAT images") from exc
    return pq


def parse_parquet_uri(uri: str) -> tuple[Path, int]:
    prefix = "parquet://"
    if not uri.startswith(prefix) or "#row=" not in uri:
        raise ValueError(f"Unsupported image source URI: {uri}")
    path, row = uri[len(prefix):].rsplit("#row=", 1)
    return Path(unquote(path)), int(row)


def image_bytes_from_source(uri: str) -> bytes:
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        path_text = unquote(parsed.path)
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
            path_text = path_text[1:]
        return Path(path_text).read_bytes()
    pq = _require_pyarrow()
    path, row_index = parse_parquet_uri(uri)
    value = pq.read_table(path, columns=["image"])["image"][row_index].as_py()
    data = (value or {}).get("bytes")
    if not data:
        raise ValueError(f"No image bytes at {uri}")
    return data


def image_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] in (b"RIFF",):
        return "image/webp"
    return "application/octet-stream"
