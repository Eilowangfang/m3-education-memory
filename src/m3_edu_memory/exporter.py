from __future__ import annotations

import html
import json
import textwrap
from io import BytesIO
from pathlib import Path

from .media import image_bytes_from_source
from .query import MemoryQuery, query_attempts
from .weakness import extract_weaknesses


def _require_media_dependencies():
    try:
        import pyarrow.parquet as pq
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("pyarrow and Pillow are required for evidence export") from exc
    return pq, Image, ImageDraw, ImageFont


def image_bytes_from_attempt(attempt: dict) -> bytes:
    return image_bytes_from_source(attempt["image_source_path"])


def render_correction_png(attempt: dict, output: Path) -> None:
    _, Image, ImageDraw, ImageFont = _require_media_dependencies()
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=24)
    lines = ["Corrected solution (reference)", "", "Question:"]
    lines.extend(textwrap.wrap(attempt["orig_q"], width=92) or [""])
    lines.extend(["", "Solution:"])
    lines.extend(textwrap.wrap(attempt["orig_a"], width=92) or [""])
    lines.extend(["", f"Evidence: {attempt['attempt_id']}"])
    width = 1400
    height = max(500, 70 + len(lines) * 27)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 28
    for index, line in enumerate(lines):
        draw.text((34, y), line, fill="black", font=title_font if index == 0 else font)
        y += 34 if index == 0 else 27
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def export_query(
    *, db_path: str | Path, query: MemoryQuery, output_dir: str | Path
) -> dict:
    result = query_attempts(db_path, query)
    result["weaknesses"] = extract_weaknesses(
        db_path, domain_code=query.domain_code
    )["weaknesses"]
    root = Path(output_dir)
    images = root / "images"
    corrections = root / "corrections"
    images.mkdir(parents=True, exist_ok=True)
    corrections.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for attempt in result["attempts"]:
        attempt_id = attempt["attempt_id"]
        image_name = f"{attempt_id}.png"
        correction_name = f"{attempt_id}-corrected.png"
        _, Image, _, _ = _require_media_dependencies()
        with Image.open(BytesIO(image_bytes_from_attempt(attempt))) as source_image:
            source_image.convert("RGB").save(images / image_name, format="PNG")
        render_correction_png(attempt, corrections / correction_name)
        cards.append(
            "<article class='card'>"
            f"<h2>{html.escape(attempt_id)}</h2>"
            f"<p><b>Time</b>: {html.escape(attempt['Time'])} (simulated)</p>"
            f"<p><b>Knowledge</b>: {html.escape(attempt['domain_code'])} / "
            f"{html.escape(attempt['subdomain_code'] or 'unknown')}</p>"
            f"<p><b>Error type</b>: {html.escape(attempt['reference_error_type'])}</p>"
            f"<p>{html.escape(attempt['pert_reasoning'] or '')}</p>"
            "<div class='pair'>"
            f"<figure><img src='images/{image_name}'><figcaption>Original handwriting</figcaption></figure>"
            f"<figure><img src='corrections/{correction_name}'><figcaption>Reference correction</figcaption></figure>"
            "</div></article>"
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    document = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>M3 Education Memory Evidence</title><style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f1;color:#202020}
main{max-width:1400px;margin:auto;padding:28px}.notice{padding:14px;background:#fff4cf;border:1px solid #d8b84a}
.card{background:white;border:1px solid #d5d5d0;margin:20px 0;padding:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
figure{margin:0}img{display:block;max-width:100%;height:auto;border:1px solid #ddd}figcaption{padding-top:6px;color:#666}
@media(max-width:850px){.pair{grid-template-columns:1fr}}</style></head><body><main>
<h1>Handwritten math error memory</h1>
<p class='notice'>This report uses FERMAT benchmark labels and simulated Time values. It does not measure VLM diagnosis accuracy.</p>
""" + "\n".join(cards) + "</main></body></html>"
    (root / "gallery.html").write_text(document, encoding="utf-8")
    return result
