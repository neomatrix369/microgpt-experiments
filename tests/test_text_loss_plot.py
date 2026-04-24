"""Verify loss text graphs: binning, min–max bands, grids, metrics, real report parse."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from run_report.parse import parse_run_report_text
from run_report.paths import run_reports_dir
from run_report.text_loss_plot import (
    _BAND,
    _D_BLOCKS,
    _MEAN,
    _delta_sparkline_symmetric,
    bin_mean,
    bin_stats,
    bin_step_ranges,
    loss_curve_comparison_lines,
    single_loss_curve_lines,
)


class TestBinMeanAndStats(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(bin_mean([], 5), [])
        self.assertEqual(bin_stats([], 5), [])

    def test_no_compression(self) -> None:
        self.assertEqual(bin_mean([1.0, 2.0, 3.0], 3), [1.0, 2.0, 3.0])
        self.assertEqual(bin_stats([1.0, 2.0, 3.0], 3), [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)])

    def test_ten_into_three_chunks(self) -> None:
        vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        stats = bin_stats(vals, 3)
        self.assertAlmostEqual(stats[0][0], 1.0)
        self.assertAlmostEqual(stats[0][1], 0.0)
        self.assertAlmostEqual(stats[0][2], 2.0)
        self.assertAlmostEqual(stats[2][0], 7.5)
        self.assertAlmostEqual(stats[2][1], 6.0)
        self.assertAlmostEqual(stats[2][2], 9.0)

    def test_thousand_into_72_first_chunk(self) -> None:
        n, w = 1000, 72
        vals = [float(i) for i in range(n)]
        b = bin_mean(vals, w)
        start = 0 * n // w
        end = 1 * n // w
        manual = sum(vals[start:end]) / (end - start)
        self.assertAlmostEqual(b[0], manual)
        self.assertEqual(len(bin_stats(vals, w)), 72)

    def test_bin_step_ranges_cover_zero_through_n_minus_one(self) -> None:
        r = bin_step_ranges(1000, 72)
        self.assertEqual(r[0][0], 0)
        self.assertEqual(r[-1][1], 999)
        self.assertEqual(len(r), 72)


class TestDeltaSymmetricSparkline(unittest.TestCase):
    def test_zero_is_center_block_on_symmetric_axis(self) -> None:
        s, lo, hi, d_scale = _delta_sparkline_symmetric([-1.0, 0.0, 1.0])
        self.assertEqual(d_scale, 1.0)
        self.assertEqual(lo, -1.0)
        self.assertEqual(hi, 1.0)
        self.assertEqual(s[0], _D_BLOCKS[0])
        self.assertEqual(s[1], _D_BLOCKS[4])
        self.assertEqual(s[2], _D_BLOCKS[7])

    def test_skewed_range_uses_max_abs_for_scale(self) -> None:
        s, lo, hi, d_scale = _delta_sparkline_symmetric([-0.04, 0.01, 0.02])
        self.assertAlmostEqual(d_scale, 0.04)
        # +0.02 maps below far right (not at █) because axis goes to +0.04
        self.assertEqual(lo, -0.04)
        self.assertEqual(hi, 0.02)


class TestGridsAndLegend(unittest.TestCase):
    def test_legend_lists_chars_and_sample_bins(self) -> None:
        lines = loss_curve_comparison_lines(
            label_a="A",
            label_b="B",
            losses_a=[3.0, 2.5, 2.0],
            losses_b=[2.8, 2.4, 2.1],
            width=3,
            grid_height=8,
        )
        blob = "\n".join(lines)
        self.assertIn("How to read:", blob)
        self.assertIn(_BAND, blob)
        self.assertIn(_MEAN, blob)
        self.assertIn("bin 0: steps 0-0", blob)
        self.assertIn("bin 1: steps 1-1", blob)

    def test_identical_runs_same_grids(self) -> None:
        losses = [3.0, 2.0, 2.5, 2.2]
        lines = loss_curve_comparison_lines(
            label_a="A",
            label_b="B",
            losses_a=losses,
            losses_b=list(losses),
            width=4,
            grid_height=8,
        )
        i_a = lines.index("  A:")
        i_b = lines.index("  B:")
        block_a = lines[i_a + 1 : i_b]
        i_metrics = next(i for i, ln in enumerate(lines) if ln.startswith("  Δ (A-B) binned means:"))
        block_b = lines[i_b + 1 : i_metrics]
        self.assertEqual(block_a, block_b)

    def test_flat_loss_single_row_band(self) -> None:
        flat = [2.5] * 8
        lines = loss_curve_comparison_lines(
            label_a="A",
            label_b="B",
            losses_a=flat,
            losses_b=list(flat),
            width=4,
            grid_height=5,
        )
        i_a = lines.index("  A:")
        grid_lines = [ln for ln in lines[i_a + 1 : i_a + 6] if "│" in ln]
        joined = "".join(grid_lines)
        self.assertNotIn(_BAND, joined)  # min=max: no vertical spread
        self.assertIn(_MEAN, joined)


class TestMetricsAndCrossParse(unittest.TestCase):
    def test_rmse_and_mean_abs_on_simple_diff(self) -> None:
        a = [0.0, 0.0, 4.0]
        b = [0.0, 3.0, 0.0]
        lines = loss_curve_comparison_lines(
            label_a="A", label_b="B", losses_a=a, losses_b=b, width=3, grid_height=8
        )
        metrics = next(ln for ln in lines if "RMSE=" in ln and "mean|Δ|" in ln)
        expect_mae = 7.0 / 3.0
        expect_rmse = math.sqrt(25.0 / 3.0)
        self.assertIn(f"RMSE={expect_rmse:.4f}", metrics)
        self.assertIn(f"mean|Δ|={expect_mae:.4f}", metrics)

    def test_real_report_if_present(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        path = next(
            (
                p
                for p in sorted(run_reports_dir(repo).glob("output_*.txt"))
                if p.is_file()
                and "--- Loss history (CSV: step,loss) ---" in p.read_text(encoding="utf-8")
            ),
            None,
        )
        if path is None:
            self.skipTest("no output_*.txt with loss history in outputs/")
        text = path.read_text(encoding="utf-8")
        p = parse_run_report_text(text)
        self.assertIsNotNone(p.loss_history)
        hist = p.loss_history
        assert hist is not None
        self.assertEqual(len(hist), int(p.config["NUM_STEPS"]))
        w = 72
        b = bin_mean(hist, w)
        self.assertEqual(len(b), w)
        edge = max(1, w // 10)
        early = sum(b[:edge]) / edge
        late = sum(b[-edge:]) / edge
        self.assertLess(
            late,
            early + 0.5,
            "binned curve should be lower at the end than the start for a typical run",
        )


class TestSingleLossCurve(unittest.TestCase):
    def test_includes_grid_with_mean_markers(self) -> None:
        losses = [3.0, 2.95, 2.8, 2.5, 2.2]
        lines = single_loss_curve_lines(
            label="run-a", losses=losses, width=5, grid_height=5
        )
        joined = "\n".join(lines)
        self.assertIn("run-a", joined)
        self.assertIn(_MEAN, joined)


if __name__ == "__main__":
    unittest.main()
