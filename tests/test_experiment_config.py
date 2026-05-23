"""Tests for RunConfig and experiment API."""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from mgpt.experiment import ExperimentResult, RunConfig, run_experiment


class TestRunConfig(unittest.TestCase):
    def test_validate_rejects_indivisible_head_split(self) -> None:
        cfg = RunConfig(n_embd=16, n_head=3)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_accepts_default(self) -> None:
        RunConfig().validate()

    def test_from_mapping_merges_fixed_and_rejects_unknown(self) -> None:
        cfg = RunConfig.from_mapping({"n_head": 1}, defaults=RunConfig())
        self.assertEqual(cfg.n_head, 1)
        self.assertEqual(cfg.n_embd, 16)
        with self.assertRaises(ValueError):
            RunConfig.from_mapping({"not_a_field": 1})

    def test_from_mapping_rejects_head_dim(self) -> None:
        with self.assertRaises(ValueError):
            RunConfig.from_mapping({"head_dim": 4})

    def test_to_cli_argv_non_default_only(self) -> None:
        cfg = RunConfig(n_head=1, num_steps=2000)
        argv = cfg.to_cli_argv()
        self.assertIn("--n-head", argv)
        self.assertIn("1", argv)
        self.assertIn("--num-steps", argv)
        self.assertIn("2000", argv)

    def test_from_argparse_namespace(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n-layer", type=int, default=1)
        parser.add_argument("--n-embd", type=int, default=16)
        parser.add_argument("--n-head", type=int, default=4)
        parser.add_argument("--block-size", type=int, default=16)
        parser.add_argument("--num-steps", type=int, default=1000)
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--learning-rate", type=float, default=0.01)
        parser.add_argument("--beta1", type=float, default=0.85)
        parser.add_argument("--beta2", type=float, default=0.99)
        parser.add_argument("--input", default="input.txt")
        parser.add_argument("--suite-index", type=int, default=2)
        args = parser.parse_args(["--n-head", "1", "--suite-index", "2"])
        cfg = RunConfig.from_argparse_namespace(args)
        self.assertEqual(cfg.n_head, 1)
        self.assertEqual(cfg.suite_index, 2)


class TestRunExperiment(unittest.TestCase):
    @patch("mgpt.experiment.generate")
    @patch("mgpt.experiment.train")
    @patch("mgpt.experiment.build_tokeniser")
    @patch("mgpt.experiment.load_dataset")
    def test_run_experiment_wires_report(
        self,
        mock_load: object,
        mock_tok: object,
        mock_train: object,
        mock_generate: object,
    ) -> None:
        mock_load.return_value = ["ann", "emma"]
        tok = type("T", (), {"vocab_size": 27, "bos": 26, "uchars": list("abcdefghijklmnopqrstuvwxyz")})()
        mock_tok.return_value = tok
        mock_train.return_value = ({}, 2.5, [3.0, 2.5])
        mock_generate.return_value = ["ann", "karia"]

        cfg = RunConfig(num_steps=2)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = run_experiment(
                cfg,
                output_dir=out,
                save_report=True,
                print_samples=False,
                print_quality=False,
            )
            self.assertIsInstance(result, ExperimentResult)
            self.assertIsNotNone(result.report_path)
            assert result.report_path is not None
            text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("OVERALL_QUALITY_SCORE=", text)
            self.assertIn("TIER1_REAL_RATIO=", text)


if __name__ == "__main__":
    unittest.main()
