"""Text-only loss curve visuals: binned min–mean–max bands + multi-row grids (stdlib only)."""

from __future__ import annotations

# Band (within-bin spread), mean marker, Δ sparkline blocks (low → high)
_BAND = "░"
_MEAN = "█"
_D_BLOCKS = "▁▂▃▄▅▆▇█"


def bin_stats(values: list[float], n_out: int) -> list[tuple[float, float, float]]:
    """Per-bin (mean, min, max) over contiguous step chunks; same chunking as :func:`bin_mean`."""
    n = len(values)
    if n_out <= 0:
        raise ValueError("n_out must be positive")
    if n == 0:
        return []
    if n_out >= n:
        return [(v, v, v) for v in values]
    out: list[tuple[float, float, float]] = []
    for i in range(n_out):
        start = i * n // n_out
        end = (i + 1) * n // n_out
        chunk = values[start:end]
        mn, mx = min(chunk), max(chunk)
        out.append((sum(chunk) / len(chunk), mn, mx))
    return out


def bin_mean(values: list[float], n_out: int) -> list[float]:
    """Compress a sequence to ``n_out`` points by averaging contiguous chunks."""
    return [m for m, _, _ in bin_stats(values, n_out)]


def bin_step_ranges(n_steps: int, n_bins: int) -> list[tuple[int, int]]:
    """Inclusive step index range per bin (matches chunk boundaries in :func:`bin_stats`)."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    ranges: list[tuple[int, int]] = []
    for i in range(n_bins):
        start = i * n_steps // n_bins
        end = (i + 1) * n_steps // n_bins - 1
        ranges.append((start, end))
    return ranges


def _value_to_row(v: float, lo: float, hi: float, height: int) -> int:
    """Map loss to row: row 0 = top = *hi*, last row = bottom = *lo*."""
    if height <= 1:
        return 0
    if hi <= lo:
        return height // 2
    t = (v - lo) / (hi - lo)
    r = round((1.0 - t) * (height - 1))
    return min(height - 1, max(0, r))


def _row_to_value(r: int, lo: float, hi: float, height: int) -> float:
    if height <= 1:
        return (lo + hi) / 2.0
    return hi - (hi - lo) * r / (height - 1)


def _render_axis_grid(
    stats: list[tuple[float, float, float]],
    lo: float,
    hi: float,
    *,
    height: int,
    label_w: int = 7,
) -> list[str]:
    """Multi-row plot: ``░`` = min–max in bin, ``█`` = mean (overwrites band)."""
    w = len(stats)
    if w == 0 or height < 1:
        return []
    cells: list[list[str]] = [[" "] * w for _ in range(height)]

    for j, (mean, mn, mx) in enumerate(stats):
        r_top = _value_to_row(mx, lo, hi, height)
        r_bot = _value_to_row(mn, lo, hi, height)
        r1, r2 = min(r_top, r_bot), max(r_top, r_bot)
        for r in range(r1, r2 + 1):
            cells[r][j] = _BAND
        r_m = _value_to_row(mean, lo, hi, height)
        cells[r_m][j] = _MEAN

    lines: list[str] = []
    for r in range(height):
        if r == 0:
            left = f"{_row_to_value(r, lo, hi, height):>{label_w}.3f}"
        elif r == height - 1:
            left = f"{_row_to_value(r, lo, hi, height):>{label_w}.3f}"
        elif r == height // 2:
            left = f"{_row_to_value(r, lo, hi, height):>{label_w}.3f}"
        else:
            left = " " * label_w
        lines.append(left + " │" + "".join(cells[r]))
    return lines


def _delta_sparkline_symmetric(diffs: list[float]) -> tuple[str, float, float, float]:
    """
    One block per bin on ±D with D = max(|Δ|), so Δ=0 is always the chart center.

    This reads better than min→max mapping when the run is skewed (e.g. mostly negative Δ
    with one small positive spike): the old scale hid zero off-center and the row looked flat.

    Returns ``(sparkline, data_min, data_max, D)``.
    """
    if not diffs:
        return "", 0.0, 0.0, 0.0
    d_lo = min(diffs)
    d_hi = max(diffs)
    d_abs = max(abs(d_lo), abs(d_hi))
    d_scale = d_abs if d_abs > 1e-15 else 1e-12
    parts: list[str] = []
    for d in diffs:
        t = (d + d_scale) / (2 * d_scale)
        idx = min(7, max(0, round(t * 7)))
        parts.append(_D_BLOCKS[idx])
    return "".join(parts), d_lo, d_hi, d_scale


def _render_delta_section(
    diffs: list[float],
    *,
    label_w: int = 7,
) -> list[str]:
    """
    Single-row Δ chart: same horizontal bins as A/B, symmetric axis so center ≈ tie.

    Left / low blocks ⇒ A's mean loss **lower** in that bin (negative Δ); right / high ⇒ A higher.
    """
    if not diffs:
        return []
    spark, span_lo, span_hi, d_scale = _delta_sparkline_symmetric(diffs)
    lw = max(label_w, len(f"{-d_scale:+.4f}"), len(f"{d_scale:+.4f}"))
    lines: list[str] = [
        "    Δ = (binned mean A - mean B). Axis is ±D with "
        "D = max(|min Δ|, |max Δ|) so the middle of the row ≈ Δ=0.",
        f"    Blocks {_D_BLOCKS}: low = A doing better there, high = A doing worse (same as lower vs higher loss).",
        f"  {f'{-d_scale:+.4f}':>{lw}} │{spark}│ {f'{d_scale:+.4f}':<{lw}}",
        f"  {'':>{lw}}   (binned Δ actually spanned {span_lo:+.4f} … {span_hi:+.4f})",
    ]
    return lines


def _format_sample_bins(ranges: list[tuple[int, int]], n_bins: int) -> str:
    if not ranges:
        return ""
    idxs = sorted({0, n_bins // 2, n_bins - 1})
    parts: list[str] = []
    for i in idxs:
        a, b = ranges[i]
        parts.append(f"bin {i}: steps {a}-{b}")
    return "; ".join(parts)


def loss_curve_comparison_lines(
    *,
    label_a: str,
    label_b: str,
    losses_a: list[float],
    losses_b: list[float],
    width: int = 72,
    grid_height: int = 12,
) -> list[str]:
    """
    Printable loss comparison: legend, shared-scale min–mean–max grids, Δ grid + scalars.

    ``width`` = number of horizontal bins. ``grid_height`` = vertical resolution (rows).
    """
    if width <= 0:
        raise ValueError("width must be positive")
    if grid_height < 3:
        raise ValueError("grid_height must be at least 3 (axis labels)")

    note: str | None = None
    if len(losses_a) != len(losses_b):
        n = min(len(losses_a), len(losses_b))
        if n == 0:
            return ["--- Loss history (text graph) ---", "  (no overlapping steps to compare)"]
        losses_a = losses_a[:n]
        losses_b = losses_b[:n]
        note = f"  (truncated to first {n} optimizer steps; runs differ in length)"

    n_steps = len(losses_a)
    ranges = bin_step_ranges(n_steps, width)
    stats_a = bin_stats(losses_a, width)
    stats_b = bin_stats(losses_b, width)

    lo = min(mn for _, mn, _ in stats_a + stats_b)
    hi = max(mx for _, _, mx in stats_a + stats_b)
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1e-12

    diffs = [a[0] - b[0] for a, b in zip(stats_a, stats_b)]
    d_lo, d_hi = min(diffs), max(diffs)
    if d_lo == d_hi:
        d_hi = d_lo + 1e-12

    err = [d * d for d in diffs]
    rmse = (sum(err) / len(err)) ** 0.5 if err else 0.0
    mean_abs = sum(abs(d) for d in diffs) / len(diffs) if diffs else 0.0

    legend = [
        "--- Loss history (text graph) ---",
        "  How to read:",
        f"    • Horizontal: {width} bins; each bin averages one contiguous block of steps.",
        f"    • Vertical: loss increases upward (top of grid = higher loss).",
        f"    • {_BAND} = spread (min–max loss) inside that bin; {_MEAN} = mean loss in that bin.",
        "    • Δ row (below): symmetric ±D axis; center ≈ tie, left/right = who leads per bin.",
        f"  Sample bins: {_format_sample_bins(ranges, width)}",
        f"  Shared loss scale (all binned min/max): {lo:.4f} … {hi:.4f}",
    ]
    if note:
        legend.insert(1, note)

    ga = _render_axis_grid(stats_a, lo, hi, height=grid_height)
    gb = _render_axis_grid(stats_b, lo, hi, height=grid_height)
    gd = _render_delta_section(diffs, label_w=7)

    tail = [
        f"  {label_a}:",
        *[f"  {ln}" for ln in ga],
        f"  {label_b}:",
        *[f"  {ln}" for ln in gb],
        f"  Δ ({label_a}-{label_b}) binned means: {d_lo:+.4f} … {d_hi:+.4f}  "
        f"RMSE={rmse:.4f}  mean|Δ|={mean_abs:.4f}",
        f"  Δ ({label_a}-{label_b}) chart:",
        *[f"  {ln}" for ln in gd],
    ]
    return legend + tail
