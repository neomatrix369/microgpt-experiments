"""Grid expansion and sweep config loading (stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any

from mgpt.experiment import FORBIDDEN_GRID_KEYS, RunConfig
from mgpt.quality import BaselineMetrics, load_baseline_from_sweep_config


@dataclass(frozen=True)
class SweepConfig:
    name: str
    fixed: dict[str, Any]
    grid: dict[str, list[Any]]
    output_dir: Path
    baseline: BaselineMetrics
    suite_note: str | None = None


def _normalize_key(key: str) -> str:
    return key.lower()


def _check_keys(mapping: dict[str, Any], *, context: str) -> None:
    allowed = RunConfig.sweep_field_names()
    for key in mapping:
        norm = _normalize_key(key)
        if norm in FORBIDDEN_GRID_KEYS or key in FORBIDDEN_GRID_KEYS:
            raise ValueError(f"{context}: '{key}' is derived; use n_embd and n_head instead")
        if norm not in allowed:
            raise ValueError(f"{context}: unknown key '{key}'")


def expand_grid(fixed: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of list-valued grid keys merged with fixed values."""
    _check_keys(fixed, context="fixed")
    _check_keys(grid, context="grid")
    if not grid:
        return [dict(fixed)]

    normalized_fixed = {_normalize_key(k): v for k, v in fixed.items()}
    grid_keys = [_normalize_key(k) for k in grid]
    value_lists: list[list[Any]] = []
    for gk in grid_keys:
        orig_key = next(k for k in grid if _normalize_key(k) == gk)
        values = grid[orig_key]
        if not isinstance(values, list):
            raise ValueError(f"grid.{orig_key} must be a list of values")
        value_lists.append(values)

    combos: list[dict[str, Any]] = []
    for values in product(*value_lists):
        combo = dict(normalized_fixed)
        for gk, val in zip(grid_keys, values, strict=True):
            combo[gk] = val
        combos.append(combo)
    return combos


def filter_valid_combos(
    combos: list[dict[str, Any]],
    *,
    defaults: RunConfig | None = None,
) -> tuple[list[RunConfig], list[tuple[dict[str, Any], str]]]:
    """Return valid configs and dropped combos with rejection reasons."""
    valid: list[RunConfig] = []
    dropped: list[tuple[dict[str, Any], str]] = []
    for combo in combos:
        try:
            cfg = RunConfig.from_mapping(combo, defaults=defaults)
            cfg.validate()
            valid.append(cfg)
        except ValueError as exc:
            dropped.append((combo, str(exc)))
    return valid, dropped


def format_dropped_combos_summary(
    dropped: list[tuple[dict[str, Any], str]],
) -> list[str]:
    """Human-readable lines describing filtered grid combinations."""
    if not dropped:
        return []
    lines = [f"Filtered {len(dropped)} invalid combination(s):"]
    for combo, reason in dropped[:10]:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))
        lines.append(f"  - {{{parts}}} — {reason}")
    if len(dropped) > 10:
        lines.append(f"  ... and {len(dropped) - 10} more")
    return lines


def _repo_root_from_config(config_path: Path) -> Path:
    p = config_path.resolve()
    for parent in [p, *p.parents]:
        if (parent / "microgpt_updated.py").is_file():
            return parent
    return config_path.parent.parent.parent


def load_sweep_config(path: Path) -> SweepConfig:
    """Load and validate a sweep JSON config file."""
    config_path = path.expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep config must be a JSON object")

    name = str(raw.get("name", config_path.stem))
    fixed = raw.get("fixed", {})
    grid = raw.get("grid", {})
    if not isinstance(fixed, dict) or not isinstance(grid, dict):
        raise ValueError("'fixed' and 'grid' must be JSON objects")

    output_dir_raw = raw.get("output_dir", f"outputs/sweeps/{name}")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = (_repo_root_from_config(config_path) / output_dir).resolve()

    baseline = load_baseline_from_sweep_config(raw, config_path)
    suite_note = raw.get("suite_note")
    if suite_note is not None:
        suite_note = str(suite_note)

    return SweepConfig(
        name=name,
        fixed=fixed,
        grid=grid,
        output_dir=output_dir,
        baseline=baseline,
        suite_note=suite_note,
    )


def planned_run_configs(
    sweep: SweepConfig,
    *,
    defaults: RunConfig | None = None,
    max_runs: int | None = None,
    strict: bool = False,
) -> tuple[list[RunConfig], list[tuple[dict[str, Any], str]]]:
    """Expand grid, validate combos, apply suite labels."""
    combos = expand_grid(sweep.fixed, sweep.grid)
    configs, dropped = filter_valid_combos(combos, defaults=defaults)
    if strict and dropped:
        raise ValueError(
            f"{len(dropped)} invalid grid combination(s); see dry-run output for details"
        )
    if max_runs is not None and len(configs) > max_runs:
        raise ValueError(
            f"grid expands to {len(configs)} valid runs; exceeds --max-runs {max_runs}"
        )
    total = len(configs)
    note = sweep.suite_note or sweep.name
    labeled: list[RunConfig] = []
    for i, cfg in enumerate(configs, start=1):
        labeled.append(
            replace(cfg, suite_index=i, suite_total=total, suite_note=note)
        )
    return labeled, dropped


def format_dry_run_table(configs: list[RunConfig]) -> list[str]:
    """Human-readable lines listing planned runs."""
    if not configs:
        return ["No valid combinations after filtering."]
    field_names = [f.name for f in fields(RunConfig)]
    skip = {"suite_index", "suite_total", "suite_note", "names_url", "eps_adam"}
    sweep_keys = [k for k in field_names if k not in skip]
    varying = [k for k in sweep_keys if len({getattr(c, k) for c in configs}) > 1]
    cols = ["#", *varying] if varying else ["#", "n_layer", "n_embd", "n_head", "num_steps"]
    lines = ["\t".join(cols)]
    for cfg in configs:
        row = [str(cfg.suite_index or "?")]
        for col in cols[1:]:
            row.append(str(getattr(cfg, col)))
        lines.append("\t".join(row))
    return lines


def _config_sort_key(path: Path) -> tuple[int, str]:
    """Sort ``N_name.json`` configs by numeric prefix, then filename."""
    stem = path.stem
    if "_" in stem:
        prefix, _, rest = stem.partition("_")
        if prefix.isdigit():
            return int(prefix), rest
    return 9999, stem


def list_sweep_config_files(configs_dir: Path | None = None) -> list[Path]:
    """Return numbered sweep JSON paths under ``experiments/configs/`` in run order."""
    if configs_dir is None:
        configs_dir = Path(__file__).resolve().parent / "configs"
    paths = sorted(configs_dir.glob("*.json"), key=_config_sort_key)
    return paths


def format_config_catalog(configs_dir: Path | None = None) -> list[str]:
    """Human-readable catalog of numbered sweep configs."""
    paths = list_sweep_config_files(configs_dir)
    if not paths:
        return ["No sweep configs found."]

    rows: list[tuple[str, str, str, str | None]] = []
    for path in paths:
        order = path.stem.split("_", 1)[0] if "_" in path.stem else "?"
        try:
            sweep = load_sweep_config(path)
            configs, _ = planned_run_configs(sweep)
            n_runs = len(configs)
            rows.append((order, path.name, sweep.name, str(n_runs)))
        except ValueError as exc:
            rows.append((order, path.name, f"(error: {exc})", None))

    file_w = max(len("File"), *(len(r[1]) for r in rows))
    name_w = max(len("Name"), *(len(r[2]) for r in rows))
    header = (
        f"{'Order':>5}  {'File':<{file_w}}  {'Name':<{name_w}}  {'Runs':>6}"
    )
    lines = [header, "-" * len(header)]
    for order, fname, name, n_runs in rows:
        if n_runs is not None:
            lines.append(
                f"{order:>5}  {fname:<{file_w}}  {name:<{name_w}}  {n_runs:>6}"
            )
        else:
            lines.append(f"{order:>5}  {fname:<{file_w}}  {name}")
    return lines
