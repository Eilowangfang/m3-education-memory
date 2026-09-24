from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .annotations import get_attempt_image
from .database import connect, initialize


RED = "#C62828"
GREEN = "#1B5E20"
INK = "#202124"
MUTED = "#5F6368"
PANEL = "#F8F9FA"
AUDIT_RENDER_VERSION = "v8-readable-inline-math"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _math_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/cambria.ttc"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return _font(size)


_SUPERSCRIPT = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "x": "ˣ", "y": "ʸ",
})


def _replace_latex_fractions(value: str) -> str:
    """Convert simple LaTeX fractions to an unambiguous one-line form."""
    while "\\frac" in value:
        start = value.find("\\frac")
        cursor = start + len("\\frac")
        groups: list[str] = []
        end = cursor
        for _ in range(2):
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor >= len(value) or value[cursor] != "{":
                return value.replace("\\frac", "")
            depth = 1
            group_start = cursor + 1
            cursor += 1
            while cursor < len(value) and depth:
                depth += (value[cursor] == "{") - (value[cursor] == "}")
                cursor += 1
            if depth:
                return value.replace("\\frac", "")
            groups.append(value[group_start:cursor - 1])
            end = cursor
        numerator = _replace_latex_fractions(groups[0]).strip()
        denominator = _replace_latex_fractions(groups[1]).strip()
        simple = re.compile(r"^[A-Za-z0-9^_ ]+$")
        if simple.fullmatch(numerator) and simple.fullmatch(denominator):
            fraction = f"{numerator.replace(' ', '')}⁄{denominator.replace(' ', '')}"
        else:
            fraction = f"({numerator})⁄({denominator})"
        value = f"{value[:start]}{fraction}{value[end:]}"
    return value


def _latex_to_display_math(value: str) -> str:
    value = _replace_latex_fractions(value.strip())
    value = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", value)
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    # PIL does not provide a TeX layout engine. Preserve the meaning of common
    # inline expressions instead of flattening commands such as
    # ``\lim_{x\to1}`` into the hard-to-read ``lim_xto1``.
    value = re.sub(r"\\lim_\{([^{}]+)\}", r"lim \1", value)
    value = re.sub(r"\\lim_([A-Za-z0-9]+)", r"lim \1", value)
    replacements = {
        r"\implies": "⇒", r"\Rightarrow": "⇒", r"\quad": "  ",
        r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\,": " ",
        r"\left": "", r"\right": "", r"\pi": "π", r"\theta": "θ",
        r"\circ": "°", r"\geq": "≥", r"\leq": "≤", r"\neq": "≠",
        r"\to": "→", r"\infty": "∞", r"\sum": "Σ", r"\prod": "Π",
        r"\int": "∫", r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
        r"\ln": "ln", r"\log": "log",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)

    def superscript(match: re.Match[str]) -> str:
        raw = match.group(1)
        if not re.fullmatch(r"[0-9+\-=()]+", raw):
            return f"^({raw.replace('-', '−')})"
        converted = raw.translate(_SUPERSCRIPT)
        return converted if len(converted) == len(raw) else f"^({raw})"

    value = re.sub(r"\^\{([^{}]+)\}", superscript, value)
    value = re.sub(r"\^([0-9nixy+\-=])", superscript, value)
    value = re.sub(r"_\{([^{}]+)\}", r"₍\1₎", value)
    value = re.sub(r"_([A-Za-z0-9])", r"₍\1₎", value)
    value = value.replace("{", "").replace("}", "").replace("$", "")
    value = value.replace("\\", "").replace("-", "−")
    value = re.sub(r"\s+", " ", value).strip(" .。")
    return value


def _formula_display_text(value: str) -> str:
    """Extract final-answer formulas and format them as readable math notation."""
    value = (value or "").strip()
    explicit = re.findall(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$([^$]+)\$", value, re.S)
    formulas = [next(part for part in group if part) for group in explicit]
    prefix = re.split(r"\\\(|\\\[|\$", value, maxsplit=1)[0]
    inline = re.search(
        r"(?<![A-Za-z0-9_])([A-Za-z](?:\([^)]*\))?\s*=\s*[A-Za-z0-9\\{}^_+\-*/().]+)",
        prefix,
    )
    if inline:
        formulas.insert(0, inline.group(1))
    if not formulas:
        formulas = [value]
    rendered = [_latex_to_display_math(formula) for formula in formulas]
    return "      ".join(dict.fromkeys(part for part in rendered if part))


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        current = ""
        for token in paragraph.split(" "):
            if draw.textbbox((0, 0), token, font=font)[2] > max_width:
                if current:
                    lines.append(current)
                    current = ""
                fragment = ""
                for character in token:
                    candidate = fragment + character
                    if fragment and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                        lines.append(fragment)
                        fragment = character
                    else:
                        fragment = candidate
                current = fragment
                continue
            candidate = token if not current else f"{current} {token}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = token
        lines.append(current)
    return lines


def _draw_wrapped(draw, xy, text, font, fill, max_width, spacing=8) -> int:
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag国", font=font)[3] + spacing
    for line in _wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def render_audit_image(
    db_path: str | Path,
    *,
    attempt_id: str,
    output_path: str | Path,
    run_id: str | None = None,
    note: str | None = None,
    correction: str | None = None,
) -> dict:
    connection = connect(db_path)
    initialize(connection)
    if run_id:
        run_filter = "AND r.run_id = ?"
        params = (attempt_id, run_id)
    else:
        run_filter = ""
        params = (attempt_id,)
    row = connection.execute(
        f"""WITH correction_ranked AS (
          SELECT cv.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY cv.attempt_id,cv.model
                   ORDER BY cv.created_at DESC,cv.version_id DESC
                 ) AS row_number
          FROM correction_versions cv WHERE cv.status='active'
        ), review_ranked AS (
          SELECT v.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY v.run_id
                   ORDER BY v.created_at DESC,v.review_id DESC
                 ) AS row_number
          FROM diagnosis_reviews v
        )
        SELECT a.*, d.*, r.model, r.provider, r.prompt_version,
               r.created_at AS analyzed_at,
               COALESCE(v.has_error_override,d.has_error_pred) AS has_error_effective,
               COALESCE(v.error_type_override,d.error_type_pred) AS error_type_effective,
               CASE WHEN COALESCE(v.has_error_override,d.has_error_pred)=0
                    THEN NULL
                    ELSE COALESCE(v.first_error_step_override,d.first_error_step)
               END AS first_error_step_effective,
               v.error_bbox_override_json,
               COALESCE(v.knowledge_points_override_json,d.knowledge_points_json)
                 AS knowledge_points_effective_json,
               COALESCE(v.error_explanation_override,d.error_explanation)
                 AS error_explanation_effective,
               v.review_id,v.reviewer,
               c.correction_id,cv.version_id AS correction_version_id,
               COALESCE(cv.corrected_solution,c.corrected_solution) AS corrected_solution,
               COALESCE(cv.corrected_steps_json,c.corrected_steps_json) AS corrected_steps_json,
               COALESCE(cv.final_answer,c.final_answer) AS final_answer,
               COALESCE(cv.verification_status,c.verification_status) AS verification_status,
               COALESCE(cv.verification_method,c.verification_method) AS verification_method
        FROM attempts a
        JOIN diagnoses d ON d.attempt_id = a.attempt_id
        JOIN analysis_runs r ON r.run_id = d.run_id
        LEFT JOIN corrections c ON c.run_id = r.run_id
        LEFT JOIN review_ranked v ON v.run_id=r.run_id AND v.row_number=1
        LEFT JOIN correction_ranked cv
          ON cv.attempt_id=a.attempt_id AND cv.model=r.model AND cv.row_number=1
        WHERE a.attempt_id = ? {run_filter} AND r.status = 'completed'
        ORDER BY r.created_at DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        raise ValueError(f"No completed diagnosis found for {attempt_id}")
    correction = correction or row["corrected_solution"]
    callout_text = row["final_answer"] or correction
    corrected_steps = json.loads(row["corrected_steps_json"] or "[]")
    reasoning_parts = [
        str(step.get("explanation") or "").strip()
        for step in corrected_steps
        if str(step.get("explanation") or "").strip()
    ]
    reasoning_text = _latex_to_display_math(
        "；".join(reasoning_parts[:2]) or correction or "暂无正确思路"
    )
    error_reason = _latex_to_display_math(
        row["error_explanation_effective"] or "模型未提供明确错误原因"
    )

    steps = connection.execute(
        "SELECT * FROM diagnosis_steps WHERE run_id=? ORDER BY step_index",
        (row["run_id"],),
    ).fetchall()
    error_steps = []
    if row["has_error_effective"]:
        if row["error_bbox_override_json"]:
            error_steps = [{
                "step_index": row["first_error_step_effective"] or 0,
                "bbox_json": row["error_bbox_override_json"],
            }]
        else:
            preferred = [
                step for step in steps
                if step["step_index"] == row["first_error_step_effective"]
                and step["bbox_json"]
            ]
            error_steps = preferred or [
                step for step in steps if step["is_error"] == 1 and step["bbox_json"]
            ]

    image_bytes, _ = get_attempt_image(db_path, attempt_id=attempt_id)
    original = Image.open(BytesIO(image_bytes)).convert("RGB")
    max_image_width = 1500
    if original.width > max_image_width:
        ratio = max_image_width / original.width
        original = original.resize((max_image_width, round(original.height * ratio)), Image.Resampling.LANCZOS)

    header_height = 120
    padding = 36
    canvas_width = max(original.width + padding * 2, 1200)
    image_x = (canvas_width - original.width) // 2
    image_y = header_height
    canvas_height = header_height + original.height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#FFFFFF")
    canvas.paste(original, (image_x, image_y))
    draw = ImageDraw.Draw(canvas, "RGBA")

    title_font = _font(34, bold=True)
    body_bold = _font(25, bold=True)
    small_font = _font(19)
    badge_font = _font(20, bold=True)

    draw.text((padding, 26), "可审计错题标注图", font=title_font, fill=INK)
    draw.text(
        (padding, 75),
        f"Attempt {attempt_id}  |  Model {row['model']}  |  Confidence {row['confidence']:.2f}",
        font=small_font,
        fill=MUTED,
    )

    error_boxes: list[tuple[int, int, int, int]] = []
    for sequence, step in enumerate(error_steps, start=1):
        x1, y1, x2, y2 = json.loads(step["bbox_json"])
        box = (
            image_x + round(x1 * original.width),
            image_y + round(y1 * original.height),
            image_x + round(x2 * original.width),
            image_y + round(y2 * original.height),
        )
        error_boxes.append(box)
        draw.rectangle(box, fill=(198, 40, 40, 28), outline=RED, width=6)
        badge = (box[0], max(image_y, box[1] - 38), box[0] + 132, box[1])
        draw.rounded_rectangle(badge, radius=9, fill=RED)
        draw.text((badge[0] + 10, badge[1] + 6), f"错误步骤 {step['step_index']}", font=badge_font, fill="white")

    # Keep the corrected result inside the photographed page. Search every
    # vertical gap after an error interval: a bottom-page error may leave no
    # space below it, while a safe gap can exist between two error steps.
    if correction and error_boxes:
        image_bottom = image_y + original.height
        gap = max(6, round(original.height * 0.006))
        bottom_margin = max(8, round(original.height * 0.008))
        callout_x1 = image_x + max(28, round(original.width * 0.05))
        callout_x2 = image_x + original.width - max(28, round(original.width * 0.05))
        desired_height = max(180, min(270, round(original.height * 0.22)))

        intervals = sorted((box[1], box[3]) for box in error_boxes)
        merged: list[list[int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        candidates: list[tuple[int, int]] = []
        for index, (_, end) in enumerate(merged):
            next_start = (
                merged[index + 1][0]
                if index + 1 < len(merged)
                else image_bottom - bottom_margin
            )
            available = next_start - gap - (end + gap)
            if available >= 90:
                candidates.append((end + gap, next_start - gap))
        callout_placement = "below"
        if candidates:
            # Prefer the largest safe gap; ties go to the lower position.
            available_y1, available_y2 = max(
                candidates, key=lambda item: (item[1] - item[0], item[0])
            )
            callout_y1 = available_y1
            callout_y2 = min(callout_y1 + desired_height, available_y2)
        else:
            # A bottom-page error can leave no room underneath it. In that case
            # use the free page area immediately above the first error box.
            available_y1 = image_y + bottom_margin
            available_y2 = merged[0][0] - gap
            if available_y2 - available_y1 >= 90:
                callout_y2 = available_y2
                callout_y1 = max(available_y1, callout_y2 - desired_height)
                callout_placement = "above"
            else:
                union_x1 = min(box[0] for box in error_boxes)
                union_x2 = max(box[2] for box in error_boxes)
                horizontal = [
                    ("left", image_x + bottom_margin, union_x1 - gap),
                    (
                        "right", union_x2 + gap,
                        image_x + original.width - bottom_margin,
                    ),
                ]
                side, side_x1, side_x2 = max(
                    horizontal, key=lambda item: item[2] - item[1]
                )
                if side_x2 - side_x1 < 320:
                    raise ValueError("Not enough non-overlapping space inside the source image for correction callout")
                callout_placement = side
                callout_x1, callout_x2 = side_x1, side_x2
                side_height = min(
                    500, image_bottom - image_y - bottom_margin * 2
                )
                error_center_y = sum(
                    (box[1] + box[3]) // 2 for box in error_boxes
                ) // len(error_boxes)
                min_y = image_y + bottom_margin
                max_y = image_bottom - bottom_margin - side_height
                callout_y1 = max(min_y, min(error_center_y - side_height // 2, max_y))
                callout_y2 = callout_y1 + side_height
        draw.rounded_rectangle(
            (callout_x1, callout_y1, callout_x2, callout_y2),
            radius=14,
            fill=(255, 255, 255, 238),
            outline=GREEN,
            width=5,
        )
        callout_height = callout_y2 - callout_y1
        compact = callout_height < 170
        label_font = _font(18 if compact else 21, bold=True)
        detail_font = _font(15 if compact else 18)
        label_width = max(
            draw.textbbox((0, 0), label, font=label_font)[2]
            for label in ("错误原因", "正确思路", "正确结果")
        )
        content_x = callout_x1 + 22 + label_width + 24
        content_width = callout_x2 - content_x - 18
        row_height = (callout_height - 16) // 3

        def draw_text_row(index: int, label: str, text: str, label_color: str) -> None:
            row_y = callout_y1 + 8 + index * row_height
            draw.text((callout_x1 + 18, row_y + 5), label, font=label_font, fill=label_color)
            lines = _wrap(draw, text, detail_font, content_width)
            line_height = draw.textbbox((0, 0), "Ag国", font=detail_font)[3] + 4
            max_lines = max(1, (row_height - 8) // line_height)
            shown = lines[:max_lines]
            if len(lines) > max_lines and shown:
                shown[-1] = shown[-1].rstrip("。；， ") + "…"
            text_y = row_y + max(3, (row_height - len(shown) * line_height) // 2)
            for line in shown:
                draw.text((content_x, text_y), line, font=detail_font, fill=INK)
                text_y += line_height

        draw_text_row(0, "错误原因", error_reason, RED)
        draw_text_row(1, "正确思路", reasoning_text, GREEN)
        for separator_index in (1, 2):
            separator_y = callout_y1 + 8 + separator_index * row_height
            draw.line(
                (callout_x1 + 14, separator_y, callout_x2 - 14, separator_y),
                fill=(27, 94, 32, 55), width=1,
            )

        result_y = callout_y1 + 8 + 2 * row_height
        draw.text((callout_x1 + 18, result_y + 5), "正确结果", font=label_font, fill=GREEN)
        formula = _formula_display_text(callout_text)
        font_factory = _font if re.search(r"[\u3400-\u9fff]", formula) else _math_font
        fitted_font = font_factory(25 if compact else 29)
        for font_size in range(25 if compact else 29, 13, -1):
            candidate_font = font_factory(font_size)
            if draw.textbbox((0, 0), formula, font=candidate_font)[2] <= content_width:
                fitted_font = candidate_font
                break
        formula_box = draw.textbbox((0, 0), formula, font=fitted_font)
        correction_y = result_y + (
            row_height - (formula_box[3] - formula_box[1])
        ) // 2 - formula_box[1]
        draw.text((content_x, correction_y), formula, font=fitted_font, fill=GREEN)
        source_box = (
            min(error_boxes, key=lambda box: box[1])
            if callout_placement == "above"
            else min(error_boxes, key=lambda box: box[0])
            if callout_placement == "left"
            else max(error_boxes, key=lambda box: box[2])
            if callout_placement == "right"
            else max(
                (box for box in error_boxes if box[3] < callout_y1),
                key=lambda box: box[3],
            )
        )
        if callout_placement in {"left", "right"}:
            source_y = (source_box[1] + source_box[3]) // 2
            draw.line(
                (
                    source_box[0] if callout_placement == "left" else source_box[2],
                    source_y,
                    callout_x2 if callout_placement == "left" else callout_x1,
                    source_y,
                ),
                fill=GREEN, width=5,
            )
        else:
            source_x = (source_box[0] + source_box[2]) // 2
            draw.line(
                (
                    source_x,
                    source_box[1] if callout_placement == "above" else source_box[3],
                    source_x,
                    callout_y2 if callout_placement == "above" else callout_y1,
                ),
                fill=GREEN, width=5,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    if row["correction_version_id"]:
        with connection:
            connection.execute(
                "UPDATE correction_versions SET rendered_path=? WHERE version_id=?",
                (str(output.resolve()), row["correction_version_id"]),
            )
    elif row["correction_id"]:
        with connection:
            connection.execute(
                "UPDATE corrections SET rendered_path=? WHERE correction_id=?",
                (str(output.resolve()), row["correction_id"]),
            )
    metadata = {
        "attempt_id": attempt_id,
        "run_id": row["run_id"],
        "model": row["model"],
        "prompt_version": row["prompt_version"],
        "image_sha256": row["image_sha256"],
        "error_step_indices": [step["step_index"] for step in error_steps],
        "display_source": "teacher_override" if row["review_id"] else "vlm",
        "review_id": row["review_id"],
        "render_version": AUDIT_RENDER_VERSION,
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    connection.close()
    return metadata
