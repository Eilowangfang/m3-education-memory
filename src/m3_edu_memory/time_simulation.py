from __future__ import annotations

import calendar
import hashlib
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


GRADE_TO_MONTH = {
    "c07": 1,
    "c08": 2,
    "c09": 3,
    "c10": 4,
    "c11": 5,
    "c12": 6,
    "c00": 7,
}

GRADE_TO_MONTH_OFFSET = {
    "c07": -6,
    "c08": -5,
    "c09": -4,
    "c10": -3,
    "c11": -2,
    "c12": -1,
    "c00": 0,
}

SIMULATION_SEED_VERSION = "v2-rolling-seven-months"


def _shift_month(anchor: date, offset: int) -> tuple[int, int]:
    month_index = anchor.year * 12 + anchor.month - 1 + offset
    return month_index // 12, month_index % 12 + 1


def simulated_time(
    *,
    grade: str,
    dataset_revision: str,
    shard: str,
    row_index: int,
    anchor_date: date | str | None = None,
    year: int | None = None,
    timezone: str = "Asia/Shanghai",
    seed: str = "m3-education-memory",
) -> datetime:
    """Return a deterministic timestamp in one of seven months ending at anchor."""
    try:
        month_offset = GRADE_TO_MONTH_OFFSET[grade]
    except KeyError as exc:
        raise ValueError(f"Unsupported FERMAT grade: {grade!r}") from exc

    # ``year`` preserves the original January-July mapping for callers that
    # explicitly request it. New imports use a rolling seven-month anchor.
    if anchor_date is None:
        anchor = date(year, 7, 31) if year is not None else date.today()
    elif isinstance(anchor_date, str):
        anchor = date.fromisoformat(anchor_date)
    else:
        anchor = anchor_date

    tz = ZoneInfo(timezone)
    target_year, target_month = _shift_month(anchor, month_offset)
    start = datetime(target_year, target_month, 1, tzinfo=tz)
    days = calendar.monthrange(target_year, target_month)[1]
    if target_year == anchor.year and target_month == anchor.month:
        days = anchor.day
    seconds = days * 24 * 60 * 60
    material = (
        f"{SIMULATION_SEED_VERSION}|{seed}|{dataset_revision}|"
        f"{shard}|{row_index}"
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % seconds
    return start + timedelta(seconds=offset)
