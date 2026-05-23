#!/usr/bin/env python3
"""Grid-search runner: JSON config → serial training runs → ranked quality summary."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mgpt.experiment import ExperimentResult, RunConfig, run_experiment
from mgpt.quality import (
    COMPARISON_METRIC_KEYS,
    BaselineMetrics,
    SWEEP_RANKING_METRIC,
    extract_comparison_metrics,
    format_baseline_comparison_lines,
    ranking_score,
    sweep_delta_csv_columns,
)
from run_report.parse import parse_run_report_text
from run_report.timing import (
    RunTiming,
    build_sweep_timing,
    capture_now,
    format_duration_seconds,
    format_progress_elapsed_eta,
    read_sweep_timing,
    write_sweep_timing,
)

from experiments.sweep_grid import (
    SweepConfig,
    format_config_catalog,
    format_dry_run_table,
    load_sweep_config,
    planned_run_configs,
)

SUMMARY_CSV = "sweep_summary.csv"
SWEEP_TIMING_TXT = "sweep_timing.txt"

SWEEP_CONFIG_COLS = (
    "n_layer",
    "n_embd",
    "n_head",
    "block_size",
    "num_steps",
    "temperature",
    "seed",
    "learning_rate",
)

QUALITY_COLS = (*COMPARISON_METRIC_KEYS, "final_loss")

DELTA_COLS = sweep_delta_csv_columns()

TIMING_COLS = (
    "started_utc",
    "started_local",
    "ended_utc",
    "ended_local",
    "duration_seconds",
    "timezone",
)


@dataclass
class SweepRow:
    suite_index: int
    config: RunConfig
    report_path: str
    final_loss: float
    semantic: dict[str, object]
    deltas: dict[str, float]
    timing: RunTiming | None = None

    def to_csv_dict(self) -> dict[str, str | float | int]:
        row: dict[str, str | float | int] = {"suite_index": self.suite_index}
        for col in SWEEP_CONFIG_COLS:
            row[col] = getattr(self.config, col)
        row["report_file"] = self.report_path
        row["final_loss"] = self.final_loss
        for col in QUALITY_COLS:
            if col == "final_loss":
                continue
            row[col] = float(self.semantic[col])
        for col in DELTA_COLS:
            key = col.removeprefix("delta_")
            row[col] = self.deltas[key]
        if self.timing is not None:
            row["started_utc"] = self.timing.started_utc
            row["started_local"] = self.timing.started_local
            row["ended_utc"] = self.timing.ended_utc
            row["ended_local"] = self.timing.ended_local
            row["duration_seconds"] = self.timing.duration_seconds
            row["timezone"] = self.timing.timezone
        else:
            for col in TIMING_COLS:
                row[col] = ""
        return row


def _csv_fieldnames() -> list[str]:
    return [
        "suite_index",
        *SWEEP_CONFIG_COLS,
        "report_file",
        *QUALITY_COLS,
        *TIMING_COLS,
        *DELTA_COLS,
    ]


def _row_from_result(
    result: ExperimentResult,
    baseline: BaselineMetrics,
) -> SweepRow:
    sem = result.semantic_quality
    deltas = baseline.delta(sem)
    report_name = result.report_path.name if result.report_path else ""
    return SweepRow(
        suite_index=int(result.config.suite_index or 0),
        config=result.config,
        report_path=report_name,
        final_loss=result.final_loss,
        semantic=extract_comparison_metrics(sem),
        deltas=deltas,
        timing=result.run_timing,
    )


def _print_baseline_comparison(row: SweepRow, baseline: BaselineMetrics) -> None:
    for line in format_baseline_comparison_lines(row.semantic, baseline):
        print(line)


def write_summary_csv(rows: list[SweepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())


def read_summary_csv(path: Path) -> list[SweepRow]:
    if not path.is_file():
        return []
    rows: list[SweepRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            cfg = RunConfig(
                n_layer=int(raw["n_layer"]),
                n_embd=int(raw["n_embd"]),
                n_head=int(raw["n_head"]),
                block_size=int(raw["block_size"]),
                num_steps=int(raw["num_steps"]),
                temperature=float(raw["temperature"]),
                seed=int(raw["seed"]),
                learning_rate=float(raw["learning_rate"]),
                suite_index=int(raw["suite_index"]),
            )
            sem = {
                key: float(raw[key])
                for key in COMPARISON_METRIC_KEYS
            }
            deltas = {k.removeprefix("delta_"): float(raw[k]) for k in DELTA_COLS}
            timing: RunTiming | None = None
            if raw.get("started_utc"):
                timing = RunTiming(
                    started_utc=str(raw.get("started_utc", "")),
                    started_local=str(raw.get("started_local", "")),
                    ended_utc=str(raw.get("ended_utc", "")),
                    ended_local=str(raw.get("ended_local", "")),
                    duration_seconds=float(raw.get("duration_seconds") or 0.0),
                    timezone=str(raw.get("timezone", "")),
                )
            rows.append(
                SweepRow(
                    suite_index=int(raw["suite_index"]),
                    config=cfg,
                    report_path=str(raw.get("report_file", "")),
                    final_loss=float(raw["final_loss"]),
                    semantic=sem,
                    deltas=deltas,
                    timing=timing,
                )
            )
    return rows


def _rows_from_output_dir(sweep: SweepConfig) -> list[SweepRow]:
    """Rebuild summary rows by parsing reports in the sweep output directory."""
    rows: list[SweepRow] = []
    reports = sorted(sweep.output_dir.glob("output_*.txt"))
    for path in reports:
        parsed = parse_run_report_text(path.read_text(encoding="utf-8"), report_filename=path.name)
        sem = parsed.semantic_quality
        if sem is None:
            continue
        cfg_raw = parsed.config
        cfg = RunConfig(
            n_layer=int(cfg_raw.get("N_LAYER", 1)),
            n_embd=int(cfg_raw.get("N_EMBD", 16)),
            n_head=int(cfg_raw.get("N_HEAD", 4)),
            block_size=int(cfg_raw.get("BLOCK_SIZE", 16)),
            num_steps=int(cfg_raw.get("NUM_STEPS", 1000)),
            temperature=float(cfg_raw.get("TEMPERATURE", 0.5)),
            seed=int(cfg_raw.get("SEED", 42)),
            learning_rate=float(cfg_raw.get("LEARNING_RATE", 0.01)),
        )
        deltas = sweep.baseline.delta(sem)
        rows.append(
            SweepRow(
                suite_index=len(rows) + 1,
                config=cfg,
                report_path=path.name,
                final_loss=parsed.final_loss,
                semantic=extract_comparison_metrics(sem),
                deltas=deltas,
                timing=parsed.run_timing,
            )
        )
    return rows


def print_ranked_table(rows: list[SweepRow], baseline: BaselineMetrics) -> None:
    if not rows:
        print("No sweep results to rank.")
        return
    ranked = sorted(
        rows,
        key=lambda r: ranking_score(r.semantic),
        reverse=True,
    )
    print("\n" + "=" * 72)
    print(f"SWEEP RANKING (by {SWEEP_RANKING_METRIC})")
    print("=" * 72)
    header = (
        f"{'Rank':<5} {'OVERALL':<9} {'Δ base':<9} {'SEC':<8} "
        f"{'N_HEAD':<7} {'STEPS':<7} report"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(ranked, start=1):
        overall = ranking_score(row.semantic)
        delta = row.deltas[SWEEP_RANKING_METRIC]
        duration = (
            f"{row.timing.duration_seconds:.1f}"
            if row.timing is not None
            else "—"
        )
        print(
            f"{rank:<5} {overall:<9.4f} {delta:+.4f}    "
            f"{duration:<8} {row.config.n_head:<7} {row.config.num_steps:<7} "
            f"{row.report_path}"
        )
    best = ranked[0]
    cli = " ".join(best.config.to_cli_argv()) or "(defaults)"
    best_delta = best.deltas[SWEEP_RANKING_METRIC]
    sign = "+" if best_delta >= 0 else ""
    print(
        f"\nBest: {cli}  "
        f"(OVERALL {ranking_score(best.semantic):.4f}, "
        f"{sign}{best_delta:.4f} vs baseline {baseline.overall_quality_score:.4f})"
    )


def maybe_generate_html(sweep: SweepConfig) -> None:
    reports = sorted(sweep.output_dir.glob("output_*.txt"))
    if len(reports) < 1:
        return
    out_html = sweep.output_dir / "comparison_report.html"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "experiments" / "report_generator.py"),
        *[str(p) for p in reports],
        "-o",
        str(out_html),
    ]
    subprocess.run(cmd, check=False, cwd=_REPO_ROOT)
    if out_html.is_file():
        print(f"\nHTML comparison: {out_html.resolve()}")


def _print_sweep_timing_summary(output_dir: Path) -> None:
    timing = read_sweep_timing(output_dir / SWEEP_TIMING_TXT)
    if timing is None:
        return
    print(
        f"Sweep timing ({timing.run_count} runs): "
        f"{format_duration_seconds(timing.duration_seconds)} "
        f"({timing.started_utc} → {timing.ended_utc}; "
        f"local {timing.started_local} → {timing.ended_local})"
    )


def run_sweep(
    sweep: SweepConfig,
    *,
    dry_run: bool = False,
    summarize_only: bool = False,
    max_runs: int | None = None,
    html: bool = False,
) -> int:
    configs = planned_run_configs(sweep, max_runs=max_runs)
    print(f"Sweep: {sweep.name}")
    print(f"Output: {sweep.output_dir}")
    print(f"Valid runs: {len(configs)}")
    print(
        f"Baseline {sweep.baseline.format_summary()}"
    )

    if dry_run:
        print("\nDry run — planned combinations:")
        for line in format_dry_run_table(configs):
            print(line)
        return 0

    summary_path = sweep.output_dir / SUMMARY_CSV
    rows: list[SweepRow] = []

    if summarize_only:
        rows = read_summary_csv(summary_path)
        if not rows:
            rows = _rows_from_output_dir(sweep)
        _print_sweep_timing_summary(sweep.output_dir)
        print_ranked_table(rows, sweep.baseline)
        if html:
            maybe_generate_html(sweep)
        return 0 if rows else 1

    sweep.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_started = capture_now()
    total_runs = len(configs)
    for run_idx, cfg in enumerate(configs):
        idx = cfg.suite_index or (run_idx + 1)
        total = cfg.suite_total or total_runs
        sweep_progress = format_progress_elapsed_eta(
            sweep_started,
            completed=run_idx,
            total=total_runs,
        )
        print(f"\n--- Run {idx} / {total} --- ({sweep_progress})")
        print(f"Config: {cfg.to_cli_argv() or '(defaults)'}")
        result = run_experiment(cfg, output_dir=sweep.output_dir, save_report=True)
        row = _row_from_result(result, sweep.baseline)
        rows.append(row)
        _print_baseline_comparison(row, sweep.baseline)
        done_progress = format_progress_elapsed_eta(
            sweep_started,
            completed=run_idx + 1,
            total=total_runs,
        )
        print(f"Sweep progress: {done_progress}")

    sweep_ended = capture_now()
    sweep_timing = build_sweep_timing(
        sweep_started,
        sweep_ended,
        run_count=len(rows),
    )
    write_summary_csv(rows, summary_path)
    write_sweep_timing(sweep.output_dir / SWEEP_TIMING_TXT, sweep_timing)
    print(f"\nSummary CSV: {summary_path.resolve()}")
    print(f"Sweep timing: {(sweep.output_dir / SWEEP_TIMING_TXT).resolve()}")
    print(
        f"Sweep wall clock: {format_duration_seconds(sweep_timing.duration_seconds)} "
        f"({sweep_timing.started_utc} → {sweep_timing.ended_utc})"
    )
    _print_sweep_timing_summary(sweep.output_dir)
    print_ranked_table(rows, sweep.baseline)
    if html:
        maybe_generate_html(sweep)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search microGPT hyperparameters by "
            f"{SWEEP_RANKING_METRIC} (see mgpt/quality.py)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Numbered JSON sweep config (see experiments/configs/1_sweep-minimal.json …)",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print numbered sweep configs in recommended run order and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List valid grid combinations without training",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Re-rank existing sweep_summary.csv or output_*.txt reports",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        metavar="N",
        help="Abort if the grid expands to more than N valid runs",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="After sweep, build comparison_report.html via report_generator.py",
    )
    args = parser.parse_args(argv)
    if args.list_configs:
        for line in format_config_catalog():
            print(line)
        return 0
    if args.config is None:
        parser.error("--config is required (or pass --list-configs)")
    try:
        sweep = load_sweep_config(args.config)
        return run_sweep(
            sweep,
            dry_run=args.dry_run,
            summarize_only=args.summarize_only,
            max_runs=args.max_runs,
            html=args.html,
        )
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
