"""Tests for mgpt.quality (baseline compare and ranking)."""

from __future__ import annotations

import unittest

from mgpt.quality import (
    BaselineMetrics,
    COMPARISON_METRIC_KEYS,
    DEFAULT_REFERENCE_BASELINE,
    SWEEP_RANKING_METRIC,
    compute_overall_quality_score,
    extract_comparison_metrics,
    format_baseline_comparison_lines,
    ranking_score,
)


class TestQuality(unittest.TestCase):
    def test_overall_formula_matches_reference(self) -> None:
        score = compute_overall_quality_score(0.55, 0.45, 0.0)
        self.assertAlmostEqual(score, DEFAULT_REFERENCE_BASELINE["overall_quality_score"])

    def test_overall_clamped(self) -> None:
        self.assertLessEqual(compute_overall_quality_score(1.0, 1.0, 0.0), 1.0)
        self.assertGreaterEqual(compute_overall_quality_score(0.0, 0.0, 1.0), 0.0)

    def test_baseline_delta(self) -> None:
        baseline = BaselineMetrics.reference()
        sem = {
            "overall_quality_score": 0.6125,
            "tier1_real_ratio": 0.6,
            "tier2_plausible_ratio": 0.4,
            "tier3_nonsense_ratio": 0.0,
        }
        deltas = baseline.delta(sem)
        self.assertAlmostEqual(deltas["overall_quality_score"], 0.03)

    def test_extract_comparison_metrics_keys(self) -> None:
        sem = {
            "tier1_real_count": 1,
            "tier1_real_ratio": 0.5,
            "tier2_plausible_ratio": 0.3,
            "tier3_nonsense_ratio": 0.2,
            "overall_quality_score": 0.4,
        }
        out = extract_comparison_metrics(sem)
        self.assertEqual(set(out.keys()), set(COMPARISON_METRIC_KEYS))

    def test_ranking_score_default_metric(self) -> None:
        sem = {"overall_quality_score": 0.75}
        self.assertEqual(ranking_score(sem), 0.75)
        self.assertEqual(SWEEP_RANKING_METRIC, "overall_quality_score")

    def test_format_baseline_comparison_lines(self) -> None:
        lines = format_baseline_comparison_lines(
            DEFAULT_REFERENCE_BASELINE,
            BaselineMetrics.reference(),
        )
        self.assertEqual(len(lines), 2)
        self.assertIn("OVERALL", lines[0])


if __name__ == "__main__":
    unittest.main()
