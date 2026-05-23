"""Tests for sweep grid expansion and config loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.sweep_grid import (
    expand_grid,
    filter_valid_combos,
    load_sweep_config,
    planned_run_configs,
)
from mgpt.quality import BaselineMetrics
from mgpt.experiment import RunConfig

_REPO = Path(__file__).resolve().parent.parent


class TestSweepGrid(unittest.TestCase):
    def test_expand_grid_cartesian_product(self) -> None:
        combos = expand_grid(
            {"n_embd": 16, "seed": 42},
            {"n_head": [1, 4], "num_steps": [1000, 2000]},
        )
        self.assertEqual(len(combos), 4)
        heads = {c["n_head"] for c in combos}
        steps = {c["num_steps"] for c in combos}
        self.assertEqual(heads, {1, 4})
        self.assertEqual(steps, {1000, 2000})

    def test_filter_drops_invalid_head_split(self) -> None:
        combos = expand_grid({"n_embd": 16}, {"n_head": [3, 4]})
        valid = filter_valid_combos(combos)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].n_head, 4)

    def test_baseline_delta(self) -> None:
        baseline = BaselineMetrics(
            overall_quality_score=0.5825,
            tier1_real_ratio=0.55,
            tier2_plausible_ratio=0.45,
            tier3_nonsense_ratio=0.0,
        )
        deltas = baseline.delta(
            {
                "overall_quality_score": 0.6125,
                "tier1_real_ratio": 0.6,
                "tier2_plausible_ratio": 0.4,
                "tier3_nonsense_ratio": 0.0,
            }
        )
        self.assertAlmostEqual(deltas["overall_quality_score"], 0.03)

    def test_load_minimal_config(self) -> None:
        path = _REPO / "experiments" / "configs" / "1_sweep-minimal.json"
        sweep = load_sweep_config(path)
        self.assertEqual(sweep.name, "minimal")
        self.assertEqual(str(sweep.output_dir).endswith("outputs/sweeps/1-minimal"), True)
        configs = planned_run_configs(sweep)
        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0].suite_index, 1)
        self.assertEqual(configs[0].suite_total, 4)

    def test_list_sweep_configs_in_order(self) -> None:
        from experiments.sweep_grid import list_sweep_config_files

        paths = list_sweep_config_files(_REPO / "experiments" / "configs")
        names = [p.name for p in paths]
        self.assertEqual(
            names,
            [
                "0_sweep-smoke-test.json",
                "1_sweep-minimal.json",
                "2_sweep-arch.json",
                "3_sweep-arch-steps.json",
                "4_sweep-full.json",
            ],
        )

    def test_format_config_catalog(self) -> None:
        from experiments.sweep_grid import format_config_catalog

        lines = format_config_catalog(_REPO / "experiments" / "configs")
        self.assertEqual(lines[0].split()[0], "Order")
        self.assertTrue(all(lines[1].strip("-") == "" for c in lines[1]))
        joined = "\n".join(lines)
        self.assertIn("0_sweep-smoke-test.json", joined)
        self.assertIn("1_sweep-minimal.json", joined)
        # Fixed-width columns: each data row has the same width as the header.
        header_len = len(lines[0])
        for line in lines[2:]:
            self.assertEqual(len(line), header_len)

    def test_unknown_fixed_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expand_grid({"bogus": 1}, {"n_head": [1]})

    def test_max_runs_enforced(self) -> None:
        path = _REPO / "experiments" / "configs" / "1_sweep-minimal.json"
        sweep = load_sweep_config(path)
        with self.assertRaises(ValueError):
            planned_run_configs(sweep, max_runs=2)

    def test_baseline_report_loading(self) -> None:
        examples = list((_REPO / "example-experiments").glob("output_*.txt"))
        if not examples:
            self.skipTest("no example-experiments reports")
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "name": "from-report",
                        "baseline_report": str(examples[0].resolve()),
                        "fixed": {},
                        "grid": {"n_head": [4]},
                    }
                ),
                encoding="utf-8",
            )
            sweep = load_sweep_config(cfg_path)
            self.assertGreater(sweep.baseline.overall_quality_score, 0.0)


if __name__ == "__main__":
    unittest.main()
