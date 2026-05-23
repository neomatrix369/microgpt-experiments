"""Wall-clock run timing with UTC and local timezone-aware ISO timestamps."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RUN_TIMING_SECTION = "--- Run timing ---"
SWEEP_TIMING_SECTION = "--- Sweep timing ---"

_FILENAME_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.txt$")

# Width for ``\\r`` training progress lines (pad so ETA is not clipped by leftovers).
CARRIAGE_PROGRESS_WIDTH = 128

_TIMING_FLOAT_KEYS = frozenset({"DURATION_SECONDS", "SWEEP_DURATION_SECONDS"})
_TIMING_STR_KEYS = frozenset(
    {
        "RUN_STARTED_UTC",
        "RUN_STARTED_LOCAL",
        "RUN_ENDED_UTC",
        "RUN_ENDED_LOCAL",
        "TIMEZONE",
        "SWEEP_STARTED_UTC",
        "SWEEP_STARTED_LOCAL",
        "SWEEP_ENDED_UTC",
        "SWEEP_ENDED_LOCAL",
        "SWEEP_TIMEZONE",
    }
)


@dataclass(frozen=True)
class RunTiming:
    """Authoritative wall-clock bounds for one train + generate + evaluate cycle."""

    started_utc: str
    started_local: str
    ended_utc: str
    ended_local: str
    duration_seconds: float
    timezone: str


@dataclass(frozen=True)
class SweepTiming:
    """Wall-clock bounds for an entire grid sweep."""

    started_utc: str
    started_local: str
    ended_utc: str
    ended_local: str
    duration_seconds: float
    timezone: str
    run_count: int


def capture_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _iso_local(dt: datetime) -> str:
    return dt.astimezone().isoformat(timespec="microseconds")


def _timezone_name(dt: datetime) -> str:
    local = dt.astimezone()
    tz = local.tzinfo
    if tz is None:
        return "UTC"
    name = tz.tzname(local)
    if name:
        return name
    return str(tz)


def build_run_timing(start: datetime, end: datetime) -> RunTiming:
    """Build timing metadata from start/end instants (any tz-aware datetimes)."""
    duration = max(0.0, (end - start).total_seconds())
    return RunTiming(
        started_utc=_iso_utc(start),
        started_local=_iso_local(start),
        ended_utc=_iso_utc(end),
        ended_local=_iso_local(end),
        duration_seconds=duration,
        timezone=_timezone_name(end),
    )


def build_sweep_timing(start: datetime, end: datetime, *, run_count: int) -> SweepTiming:
    duration = max(0.0, (end - start).total_seconds())
    return SweepTiming(
        started_utc=_iso_utc(start),
        started_local=_iso_local(start),
        ended_utc=_iso_utc(end),
        ended_local=_iso_local(end),
        duration_seconds=duration,
        timezone=_timezone_name(end),
        run_count=run_count,
    )


def format_run_timing_lines(timing: RunTiming) -> list[str]:
    return [
        RUN_TIMING_SECTION,
        f"RUN_STARTED_UTC={timing.started_utc}",
        f"RUN_STARTED_LOCAL={timing.started_local}",
        f"RUN_ENDED_UTC={timing.ended_utc}",
        f"RUN_ENDED_LOCAL={timing.ended_local}",
        f"DURATION_SECONDS={timing.duration_seconds:.6f}",
        f"TIMEZONE={timing.timezone}",
        "",
    ]


def format_sweep_timing_lines(timing: SweepTiming) -> list[str]:
    return [
        SWEEP_TIMING_SECTION,
        f"SWEEP_STARTED_UTC={timing.started_utc}",
        f"SWEEP_STARTED_LOCAL={timing.started_local}",
        f"SWEEP_ENDED_UTC={timing.ended_utc}",
        f"SWEEP_ENDED_LOCAL={timing.ended_local}",
        f"SWEEP_DURATION_SECONDS={timing.duration_seconds:.6f}",
        f"SWEEP_TIMEZONE={timing.timezone}",
        f"SWEEP_RUN_COUNT={timing.run_count}",
        "",
    ]


def write_sweep_timing(path: Path, timing: SweepTiming) -> None:
    """Write sweep timing block to ``path`` (one section, trailing newline)."""
    path.write_text("\n".join(format_sweep_timing_lines(timing)), encoding="utf-8")


def _parse_key_values_in_section(
    text: str,
    section_header: str,
) -> dict[str, str | float]:
    out: dict[str, str | float] = {}
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == section_header:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("---"):
                break
            if not stripped or "=" not in stripped:
                continue
            key, _, rest = stripped.partition("=")
            key = key.strip()
            val = rest.strip()
            if key in _TIMING_FLOAT_KEYS:
                try:
                    out[key] = float(val)
                except ValueError:
                    pass
            elif key in _TIMING_STR_KEYS:
                out[key] = val
            elif key == "SWEEP_RUN_COUNT":
                try:
                    out[key] = int(val)
                except ValueError:
                    pass
    return out


def parse_run_timing(text: str) -> RunTiming | None:
    """Parse ``--- Run timing ---`` block; return ``None`` if absent."""
    kv = _parse_key_values_in_section(text, RUN_TIMING_SECTION)
    started_utc = kv.get("RUN_STARTED_UTC")
    if not isinstance(started_utc, str) or not started_utc:
        return None
    try:
        return RunTiming(
            started_utc=str(kv.get("RUN_STARTED_UTC", "")),
            started_local=str(kv.get("RUN_STARTED_LOCAL", "")),
            ended_utc=str(kv.get("RUN_ENDED_UTC", "")),
            ended_local=str(kv.get("RUN_ENDED_LOCAL", "")),
            duration_seconds=float(kv.get("DURATION_SECONDS", 0.0)),
            timezone=str(kv.get("TIMEZONE", "")),
        )
    except (TypeError, ValueError):
        return None


def parse_sweep_timing(text: str) -> SweepTiming | None:
    """Parse ``--- Sweep timing ---`` block; return ``None`` if absent."""
    kv = _parse_key_values_in_section(text, SWEEP_TIMING_SECTION)
    started_utc = kv.get("SWEEP_STARTED_UTC")
    if not isinstance(started_utc, str) or not started_utc:
        return None
    try:
        run_count_raw = kv.get("SWEEP_RUN_COUNT", 0)
        run_count = int(run_count_raw) if isinstance(run_count_raw, (int, float)) else 0
        return SweepTiming(
            started_utc=str(kv.get("SWEEP_STARTED_UTC", "")),
            started_local=str(kv.get("SWEEP_STARTED_LOCAL", "")),
            ended_utc=str(kv.get("SWEEP_ENDED_UTC", "")),
            ended_local=str(kv.get("SWEEP_ENDED_LOCAL", "")),
            duration_seconds=float(kv.get("SWEEP_DURATION_SECONDS", 0.0)),
            timezone=str(kv.get("SWEEP_TIMEZONE", "")),
            run_count=run_count,
        )
    except (TypeError, ValueError):
        return None


def read_sweep_timing(path: Path) -> SweepTiming | None:
    """Load sweep timing sidecar written by ``write_sweep_timing``."""
    if not path.is_file():
        return None
    return parse_sweep_timing(path.read_text(encoding="utf-8"))


def parse_filename_local_timestamp(filename: str) -> str | None:
    """Legacy fallback: naive local ``YYYYMMDD_HHMMSS`` suffix from report filename."""
    m = _FILENAME_TS_RE.search(filename)
    if not m:
        return None
    date_s, time_s = m.group(1), m.group(2)
    return (
        f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}T"
        f"{time_s[:2]}:{time_s[2:4]}:{time_s[4:6]} (local, from filename)"
    )


def local_filename_timestamp(dt: datetime | None = None) -> str:
    """Filesystem suffix ``YYYYMMDD_HHMMSS`` in local wall-clock time."""
    instant = dt or capture_now()
    return instant.astimezone().strftime("%Y%m%d_%H%M%S")


def format_duration_seconds(seconds: float) -> str:
    """Human-readable duration for console progress (e.g. ``1m 23s``, ``45.2s``)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_progress_elapsed_eta(
    start: datetime,
    *,
    completed: int,
    total: int,
    now: datetime | None = None,
) -> str:
    """``elapsed … | ETA …`` for step/run progress lines."""
    if total <= 0:
        return "elapsed — | ETA —"
    instant = now or capture_now()
    elapsed_s = max(0.0, (instant - start).total_seconds())
    elapsed = format_duration_seconds(elapsed_s)
    remaining = max(0, total - completed)
    if completed <= 0 or remaining <= 0:
        eta = "0s" if remaining <= 0 else "—"
    else:
        eta_s = (elapsed_s / completed) * remaining
        eta = format_duration_seconds(eta_s)
    return f"elapsed {elapsed} | ETA {eta}"


def print_carriage_progress(
    message: str,
    *,
    width: int = CARRIAGE_PROGRESS_WIDTH,
) -> None:
    """Print an in-place progress line (``\\r``) padded so ETA is not truncated."""
    if len(message) > width:
        message = message[: width - 1] + "…"
    sys.stdout.write(message.ljust(width) + "\r")
    sys.stdout.flush()
