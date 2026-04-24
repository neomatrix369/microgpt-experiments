"""Output filename encoding for run reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Subdirectory name (under the microgpt repository root) for ``output_*.txt`` run
# reports and the default HTML comparison path. Created on write as needed.
DEFAULT_RUN_REPORT_DIR = Path("outputs")


def run_reports_dir(repo_root: Path) -> Path:
    """Return ``repo_root / DEFAULT_RUN_REPORT_DIR``.

    Pass the microgpt **repository root** (the directory that contains
    ``microgpt_updated.py``), not :func:`pathlib.Path.cwd`, so every tool agrees on
    where artifacts live regardless of the shell's current directory.
    """
    return repo_root / DEFAULT_RUN_REPORT_DIR


def format_run_output_path_for_params(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
    head_dim: int,
    block_size: int,
    num_steps: int,
    temperature: float,
    seed: int,
    prefix: str = "output",
    directory: str | Path = DEFAULT_RUN_REPORT_DIR,
) -> Path:
    """Build a filesystem-safe path from hyperparameters and current local time.

    Example: ``output_L1_E16_H4_D4_B16_S1000_T0p5_seed42_20260422_153045.txt``
    (temperature dots become ``p``; trailing ``YYYYMMDD_HHMMSS`` is local wall-clock
    time when this function runs).
    """
    t_token = f"{temperature:g}".replace("-", "m").replace(".", "p")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"{prefix}_L{n_layer}_E{n_embd}_H{n_head}_D{head_dim}_B{block_size}"
        f"_S{num_steps}_T{t_token}_seed{seed}_{run_ts}"
    )
    return Path(directory) / f"{stem}.txt"
