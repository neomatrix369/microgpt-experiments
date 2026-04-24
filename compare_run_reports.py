#!/usr/bin/env python3
"""
Compare two microGPT ``output_*.txt`` run reports.

Smallest useful diff:
  - Config (KEY=value under training) — reproducibility / sweep identity;
    when some keys differ, matching keys are listed once under *Shared config*.
  - Final loss — one training outcome scalar
  - Inference samples — ordered lines (what you actually read)

If both reports include ``--- Loss history ---``, **text graphs** are printed:
multi-row grids with per-bin min–mean–max bands, shared loss scale for A and B,
a Δ chart, and RMSE / mean |Δ| over binned means (default 72 bins × 12 rows).

Usage::

    python compare_run_reports.py output_A.txt output_B.txt
    python compare_run_reports.py output_A.txt output_B.txt --loss-bins 96 --loss-height 14

Exit codes: ``0`` if config, loss, and samples all match; ``1`` if any differ;
``2`` for bad arguments or an unreadable report. Narrative and glossary sections
are not compared. See ``README.md`` (section *Comparing two reports*)
for context.

Reports are usually under ``<repo>/outputs/`` (see ``run_report.paths.run_reports_dir``);
pass explicit paths if yours live elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_report import (
    DERIVED_EXPERIMENT_CFG_KEYS,
    cfg_keys_for_experiment_table,
    experiment_cfg_calculated_caption,
    loss_curve_comparison_lines,
    parse_run_report_text,
)


def _fmt_val(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _config_print_label(key: str) -> str:
    cap = experiment_cfg_calculated_caption(key)
    return f"{key} ({cap})" if cap else key


def _config_line(key: str, val: object) -> str:
    cap = experiment_cfg_calculated_caption(key)
    suffix = f"  — {cap}" if cap else ""
    return f"{key}={_fmt_val(val)}{suffix}"


def parse_run_report(
    text: str,
) -> tuple[dict[str, int | float | str], float, list[str], list[float] | None]:
    """Parse a run report; kept for importers. Prefer :func:`parse_run_report_text`."""
    p = parse_run_report_text(text)
    return p.config, p.final_loss, p.samples, p.loss_history


def compare_reports(path_a: Path, path_b: Path, *, loss_bins: int, loss_height: int) -> int:
    text_a = path_a.read_text(encoding="utf-8")
    text_b = path_b.read_text(encoding="utf-8")
    cfg_a, loss_a, samp_a, hist_a = parse_run_report(text_a)
    cfg_b, loss_b, samp_b, hist_b = parse_run_report(text_b)

    exit_code = 0
    keys = cfg_keys_for_experiment_table(
        (set(cfg_a) | set(cfg_b)) - DERIVED_EXPERIMENT_CFG_KEYS
    )
    config_diff: list[tuple[str, str, str]] = []
    shared_keys: set[str] = set()
    for k in keys:
        va, vb = cfg_a.get(k), cfg_b.get(k)
        if va != vb:
            exit_code = 1
            config_diff.append(
                (k, _fmt_val(va) if va is not None else "—", _fmt_val(vb) if vb is not None else "—")
            )
        else:
            shared_keys.add(k)

    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print()

    if config_diff:
        if shared_keys:
            print("--- Shared config (both runs) ---")
            for k in cfg_keys_for_experiment_table(shared_keys):
                print(f"  {_config_line(k, cfg_a[k])}")
            print()
        print("--- Config differences ---")
        prefix_w = max(len(f"{_config_print_label(k)}:  ") for k, _, _ in config_diff)
        w_a = max(len(f"A={a}") for _, a, _ in config_diff)
        for k, a, b in config_diff:
            label = _config_print_label(k)
            pad = prefix_w - len(f"{label}:  ")
            left = f"A={a}".ljust(w_a)
            print(f"  {label}:  {' ' * pad}{left}  |  B={b}")
        print()
    else:
        print("--- Config (same both runs) ---")
        display_keys = cfg_keys_for_experiment_table(set(cfg_a))
        for k in display_keys:
            print(f"  {_config_line(k, cfg_a[k])}")
        print()

    if loss_a != loss_b:
        exit_code = 1
        print("--- Final loss ---")
        print(f"  A: {loss_a:.6f}")
        print(f"  B: {loss_b:.6f}")
        print()
    else:
        print(f"--- Final loss: {loss_a:.6f} (same) ---\n")

    if hist_a is not None and hist_b is not None:
        print(
            *loss_curve_comparison_lines(
                label_a="A",
                label_b="B",
                losses_a=hist_a,
                losses_b=hist_b,
                width=loss_bins,
                grid_height=loss_height,
            ),
            sep="\n",
        )
        print()
    elif hist_a is not None or hist_b is not None:
        which = path_a.name if hist_a is not None else path_b.name
        print("--- Loss history (text graph) ---")
        print(f"  skipped: only one run has loss history (see {which})")
        print()

    if samp_a != samp_b:
        exit_code = 1
        print("--- Inference samples ---")
        n = max(len(samp_a), len(samp_b))
        pairs: list[tuple[str, str]] = [
            (
                samp_a[i] if i < len(samp_a) else "—",
                samp_b[i] if i < len(samp_b) else "—",
            )
            for i in range(n)
        ]
        w_a = max((len(la) for la, _ in pairs), default=0)
        w_b = max((len(lb) for _, lb in pairs), default=0)
        for i, (la, lb) in enumerate(pairs, start=1):
            mark = " " if la == lb else "*"
            print(f"{mark} {i:2d}:  A: {la:<{w_a}}  B: {lb:<{w_b}}")
    else:
        print(f"--- Inference samples: identical ({len(samp_a)} lines) ---")

    return exit_code


def main() -> None:
    args = sys.argv[1:]
    loss_bins = 72
    loss_height = 12
    if "--loss-bins" in args:
        i = args.index("--loss-bins")
        try:
            loss_bins = int(args[i + 1])
        except (IndexError, ValueError):
            print("Usage: ... --loss-bins N  (N positive integer)", file=sys.stderr)
            sys.exit(2)
        del args[i : i + 2]
    if "--loss-height" in args:
        i = args.index("--loss-height")
        try:
            loss_height = int(args[i + 1])
        except (IndexError, ValueError):
            print("Usage: ... --loss-height N  (N integer >= 3)", file=sys.stderr)
            sys.exit(2)
        del args[i : i + 2]
    if loss_bins < 1:
        print("--loss-bins must be at least 1", file=sys.stderr)
        sys.exit(2)
    if loss_height < 3:
        print("--loss-height must be at least 3", file=sys.stderr)
        sys.exit(2)
    if len(args) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    a, b = Path(args[0]), Path(args[1])
    if not a.is_file() or not b.is_file():
        print("Both arguments must be existing files.", file=sys.stderr)
        sys.exit(2)
    try:
        code = compare_reports(a, b, loss_bins=loss_bins, loss_height=loss_height)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
