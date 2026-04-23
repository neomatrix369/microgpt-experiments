#!/usr/bin/env python3
"""
Compare two microGPT ``output_*.txt`` run reports.

Smallest useful diff:
  - Config (KEY=value under training) — reproducibility / sweep identity;
    when some keys differ, matching keys are listed once under *Shared config*.
  - Final loss — one training outcome scalar
  - Inference samples — ordered lines (what you actually read)

Usage::

    python compare_run_reports.py output_A.txt output_B.txt

Exit codes: ``0`` if config, loss, and samples all match; ``1`` if any differ;
``2`` for bad arguments or an unreadable report. Narrative and glossary sections
are not compared. See ``README.md`` (section *Comparing two reports*) for context.
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_report import (
    cfg_keys_in_display_order,
    parse_run_report_text,
)


def _fmt_val(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def parse_run_report(text: str) -> tuple[dict[str, int | float | str], float, list[str]]:
    """Parse a run report; kept for importers. Prefer :func:`parse_run_report_text`."""
    p = parse_run_report_text(text)
    return p.config, p.final_loss, p.samples


def compare_reports(path_a: Path, path_b: Path) -> int:
    text_a = path_a.read_text(encoding="utf-8")
    text_b = path_b.read_text(encoding="utf-8")
    cfg_a, loss_a, samp_a = parse_run_report(text_a)
    cfg_b, loss_b, samp_b = parse_run_report(text_b)

    exit_code = 0
    keys = sorted(set(cfg_a) | set(cfg_b))
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
            for k in cfg_keys_in_display_order(shared_keys):
                print(f"  {k}={_fmt_val(cfg_a[k])}")
            print()
        print("--- Config differences ---")
        prefix_w = max(len(f"{k}:  ") for k, _, _ in config_diff)
        w_a = max(len(f"A={a}") for _, a, _ in config_diff)
        for k, a, b in config_diff:
            pad = prefix_w - len(f"{k}:  ")
            left = f"A={a}".ljust(w_a)
            print(f"  {k}:  {' ' * pad}{left}  |  B={b}")
        print()
    else:
        print("--- Config (same both runs) ---")
        display_keys = cfg_keys_in_display_order(set(cfg_a))
        for k in display_keys:
            print(f"  {k}={_fmt_val(cfg_a[k])}")
        print()

    if loss_a != loss_b:
        exit_code = 1
        print("--- Final loss ---")
        print(f"  A: {loss_a:.6f}")
        print(f"  B: {loss_b:.6f}")
        print()
    else:
        print(f"--- Final loss: {loss_a:.6f} (same) ---\n")

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
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    if not a.is_file() or not b.is_file():
        print("Both arguments must be existing files.", file=sys.stderr)
        sys.exit(2)
    try:
        code = compare_reports(a, b)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
