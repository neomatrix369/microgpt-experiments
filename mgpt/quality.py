"""Central hub for sample-quality metrics, baseline comparison, and sweep ranking.

All computation of tier/heuristic scores lives in :mod:`mgpt.evaluation`.
This module owns the **objective function** (overall score formula), **comparison**
(vs a baseline), and **ranking** keys used by grid sweep and docs.

To replace heuristics with better ideas, see :func:`extension_points_doc` and
the *Extending quality scoring* section in ``docs/M2-semantic-quality.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_report.parse import parse_run_report_text

# Metrics compared against a baseline and written to sweep_summary.csv
COMPARISON_METRIC_KEYS: tuple[str, ...] = (
    "overall_quality_score",
    "tier1_real_ratio",
    "tier2_plausible_ratio",
    "tier3_nonsense_ratio",
)

# Grid sweep ranks runs by this semantic dict key (change here to retarget sweep).
SWEEP_RANKING_METRIC = "overall_quality_score"

# Checked-in reference: H4 @ 1000 steps (``example-experiments/``).
DEFAULT_REFERENCE_BASELINE: dict[str, float] = {
    "overall_quality_score": 0.5825,
    "tier1_real_ratio": 0.55,
    "tier2_plausible_ratio": 0.45,
    "tier3_nonsense_ratio": 0.0,
}


def compute_overall_quality_score(
    tier1_real_ratio: float,
    tier2_plausible_ratio: float,
    tier3_nonsense_ratio: float,
) -> float:
    """Combine tier ratios into one number in [0, 1] (grid-sweep objective).

    Weights: real x 1.0, plausible x 0.7, non-nonsense x 0.3, then / 2.
    """
    overall = (
        tier1_real_ratio * 1.0
        + tier2_plausible_ratio * 0.7
        + (1.0 - tier3_nonsense_ratio) * 0.3
    ) / 2.0
    return max(0.0, min(1.0, overall))


def extract_comparison_metrics(semantic: dict[str, object]) -> dict[str, float]:
    """Subset of a semantic-quality dict used for compare, CSV, and ranking."""
    return {key: float(semantic[key]) for key in COMPARISON_METRIC_KEYS}


def ranking_score(
    semantic: dict[str, object],
    *,
    metric: str = SWEEP_RANKING_METRIC,
) -> float:
    """Scalar used to order runs (higher is better for the default metric)."""
    return float(semantic[metric])


def delta_csv_column(metric_key: str) -> str:
    return f"delta_{metric_key}"


def sweep_delta_csv_columns() -> tuple[str, ...]:
    return tuple(delta_csv_column(k) for k in COMPARISON_METRIC_KEYS)


@dataclass(frozen=True)
class BaselineMetrics:
    """Reference quality numbers for delta comparison (e.g. a known-good run)."""

    overall_quality_score: float
    tier1_real_ratio: float
    tier2_plausible_ratio: float
    tier3_nonsense_ratio: float

    @classmethod
    def reference(cls) -> BaselineMetrics:
        """Default project baseline (H4 @ 1000 steps)."""
        return cls.from_mapping(DEFAULT_REFERENCE_BASELINE)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> BaselineMetrics:
        try:
            return cls(
                overall_quality_score=float(mapping["overall_quality_score"]),
                tier1_real_ratio=float(mapping["tier1_real_ratio"]),
                tier2_plausible_ratio=float(mapping["tier2_plausible_ratio"]),
                tier3_nonsense_ratio=float(mapping["tier3_nonsense_ratio"]),
            )
        except KeyError as exc:
            raise ValueError(f"baseline missing key: {exc.args[0]}") from exc

    @classmethod
    def from_semantic(cls, semantic: dict[str, object]) -> BaselineMetrics:
        return cls.from_mapping(extract_comparison_metrics(semantic))

    def as_dict(self) -> dict[str, float]:
        return extract_comparison_metrics(
            {
                "overall_quality_score": self.overall_quality_score,
                "tier1_real_ratio": self.tier1_real_ratio,
                "tier2_plausible_ratio": self.tier2_plausible_ratio,
                "tier3_nonsense_ratio": self.tier3_nonsense_ratio,
            }
        )

    def delta(self, semantic: dict[str, object]) -> dict[str, float]:
        """Per-metric difference: run minus baseline."""
        current = extract_comparison_metrics(semantic)
        base = self.as_dict()
        return {key: current[key] - base[key] for key in COMPARISON_METRIC_KEYS}

    def format_summary(self) -> str:
        return (
            f"OVERALL={self.overall_quality_score:.4f} "
            f"(T1={self.tier1_real_ratio:.2f} "
            f"T2={self.tier2_plausible_ratio:.2f} "
            f"T3={self.tier3_nonsense_ratio:.2f})"
        )


def compare_to_baseline(
    semantic: dict[str, object],
    baseline: BaselineMetrics,
) -> dict[str, float]:
    """Alias for :meth:`BaselineMetrics.delta` (run vs reference)."""
    return baseline.delta(semantic)


def format_baseline_comparison_lines(
    semantic: dict[str, object],
    baseline: BaselineMetrics,
) -> list[str]:
    """Console lines after each sweep run (deltas vs baseline)."""
    metrics = extract_comparison_metrics(semantic)
    deltas = baseline.delta(semantic)
    overall = metrics["overall_quality_score"]
    delta = deltas["overall_quality_score"]
    sign = "+" if delta >= 0 else ""
    beat = "✓" if delta > 0 else ("=" if delta == 0 else "✗")
    return [
        (
            f"  OVERALL {overall:.4f} ({sign}{delta:.4f} vs baseline "
            f"{baseline.overall_quality_score:.4f}) {beat}"
        ),
        (
            f"  TIER1 {metrics['tier1_real_ratio']:.4f} "
            f"({deltas['tier1_real_ratio']:+.4f}) | "
            f"TIER2 {metrics['tier2_plausible_ratio']:.4f} "
            f"({deltas['tier2_plausible_ratio']:+.4f}) | "
            f"TIER3 {metrics['tier3_nonsense_ratio']:.4f} "
            f"({deltas['tier3_nonsense_ratio']:+.4f})"
        ),
    ]


def load_baseline_from_sweep_config(
    raw: dict[str, Any],
    config_path: Path,
) -> BaselineMetrics:
    """Load baseline from sweep JSON ``baseline`` or ``baseline_report`` path."""
    if "baseline_report" in raw:
        report_path = Path(raw["baseline_report"])
        if not report_path.is_absolute():
            report_path = (config_path.parent / report_path).resolve()
        text = report_path.read_text(encoding="utf-8")
        parsed = parse_run_report_text(text)
        sem = parsed.semantic_quality
        if sem is None:
            raise ValueError(f"baseline_report has no semantic quality block: {report_path}")
        return BaselineMetrics.from_semantic(sem)

    baseline = raw.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("config must include 'baseline' or 'baseline_report'")
    return BaselineMetrics.from_mapping(baseline)


def extension_points_doc() -> str:
    """Short map of where to plug in better quality ideas (for docs/tests)."""
    return (
        "Tier rules and per-sample scoring: mgpt/evaluation.py. "
        "Overall objective weights: mgpt/quality.compute_overall_quality_score. "
        "Sweep ranking metric: mgpt/quality.SWEEP_RANKING_METRIC. "
        "Report text format: run_report/builder.py and run_report/parse.py."
    )
