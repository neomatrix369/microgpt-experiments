#!/usr/bin/env python3
"""
Insert the same ``--- What this run is ---`` narrative that :func:`save_run_report`
writes into **existing** ``output_*.txt`` files (past experiments).

Stdlib only; run from the repo root. With no arguments, scans ``outputs/output_*.txt``::

    python annotate_run_reports.py
    python annotate_run_reports.py outputs/output_L1_....txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from run_report import (
    NARRATIVE_SECTION_HEADER,
    NARRATIVE_REQUIRED_CONFIG_KEYS,
    format_run_narrative_lines,
    parse_run_report_text,
)
from run_report.paths import run_reports_dir


def annotate_file(path: Path) -> str:
    """Return a status string: 'skip', 'ok', or 'err: ...'."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"err: {e}"
    if NARRATIVE_SECTION_HEADER in text:
        return "skip (narrative already present)"
    try:
        p = parse_run_report_text(text)
    except ValueError:
        return "err: could not parse config / loss (unexpected format)"
    cfg = p.config
    for k in NARRATIVE_REQUIRED_CONFIG_KEYS:
        if k not in cfg:
            return "err: could not parse config / loss (unexpected format)"
    if len(p.samples) < 1:
        return "err: no Sample N: lines found"

    n_layer = int(cfg["N_LAYER"])
    n_embd = int(cfg["N_EMBD"])
    n_head = int(cfg["N_HEAD"])
    head_dim = int(cfg["HEAD_DIM"])
    block_size = int(cfg["BLOCK_SIZE"])
    num_steps = int(cfg["NUM_STEPS"])
    temperature = float(cfg["TEMPERATURE"])
    seed = int(cfg["SEED"])
    input_path = str(cfg["INPUT_PATH"])
    final_loss = p.final_loss
    num_samples = len(p.samples)

    sep = "==================="
    i = text.find(sep)
    if i < 0:
        return "err: missing report banner line"
    insert_at = text.find("\n", i) + 1
    narrative = (
        "\n".join(
            format_run_narrative_lines(
                n_layer=n_layer,
                n_embd=n_embd,
                n_head=n_head,
                head_dim=head_dim,
                block_size=block_size,
                num_steps=num_steps,
                temperature=temperature,
                seed=seed,
                input_path=input_path,
                final_loss=final_loss,
                num_samples=num_samples,
            )
        )
        + "\n"
    )
    new_text = text[:insert_at] + narrative + text[insert_at:]
    path.write_text(new_text, encoding="utf-8")
    return "ok"


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(run_reports_dir(repo_root).glob("output_*.txt"))
    if not paths:
        print(
            f"No output_*.txt in {run_reports_dir(repo_root)} "
            "(pass explicit paths as arguments)."
        )
        sys.exit(0)
    for p in paths:
        status = annotate_file(p)
        print(f"{p.name}: {status}")


if __name__ == "__main__":
    main()
