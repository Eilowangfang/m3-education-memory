from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
from urllib.parse import unquote
from urllib.parse import urlparse


_PARQUET_CACHE_LOCK = threading.Lock()
_PARQUET_CACHE_KEY: tuple[str, int] | None = None
_PARQUET_CACHE_VALUES: list[dict | None] = []


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
    parquet = pq.ParquetFile(path)
    row_offset = 0
    row_group = None
    for index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(index).num_rows
        if row_index < row_offset + row_count:
            row_group = index
            break
        row_offset += row_count
    if row_group is None:
        raise IndexError(f"Parquet row {row_index} is out of range for {path}")

    cache_key = (str(path.resolve()), row_group)
    global _PARQUET_CACHE_KEY, _PARQUET_CACHE_VALUES
    with _PARQUET_CACHE_LOCK:
        if _PARQUET_CACHE_KEY != cache_key:
            column = parquet.read_row_group(row_group, columns=["image"])["image"]
            _PARQUET_CACHE_VALUES = column.to_pylist()
            _PARQUET_CACHE_KEY = cache_key
        value = _PARQUET_CACHE_VALUES[row_index - row_offset]
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
