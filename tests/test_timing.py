"""Tests for run timing capture, formatting, and parsing."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_report.builder import build_run_report_lines
from run_report.parse import parse_run_report_text
from run_report.timing import (
    RunTiming,
    build_run_timing,
    build_sweep_timing,
    format_duration_seconds,
    format_progress_elapsed_eta,
    format_run_timing_lines,
    parse_filename_local_timestamp,
    parse_run_timing,
    parse_sweep_timing,
    read_sweep_timing,
    write_sweep_timing,
)
from run_report.timing import CARRIAGE_PROGRESS_WIDTH


class TestRunTiming(unittest.TestCase):
    def test_build_and_format_round_trip(self) -> None:
        start = datetime(2026, 5, 23, 22, 30, 45, 123456, tzinfo=timezone.utc)
        end = start + timedelta(seconds=12.5)
        timing = build_run_timing(start, end)
        self.assertEqual(timing.started_utc, "2026-05-23T22:30:45.123456+00:00")
        self.assertAlmostEqual(timing.duration_seconds, 12.5)
        lines = format_run_timing_lines(timing)
        text = "\n".join(lines)
        parsed = parse_run_timing(text)
        assert parsed is not None
        self.assertEqual(parsed.started_utc, timing.started_utc)
        self.assertAlmostEqual(parsed.duration_seconds, timing.duration_seconds)

    def test_parse_run_report_includes_timing(self) -> None:
        timing = RunTiming(
            started_utc="2026-05-23T22:30:45.123456+00:00",
            started_local="2026-05-23T15:30:45.123456-07:00",
            ended_utc="2026-05-23T22:31:00.123456+00:00",
            ended_local="2026-05-23T15:31:00.123456-07:00",
            duration_seconds=15.0,
            timezone="PDT",
        )
        lines = build_run_report_lines(
            n_layer=1,
            n_embd=16,
            n_head=4,
            block_size=16,
            num_steps=2,
            temperature=0.5,
            seed=42,
            learning_rate=0.01,
            beta1=0.85,
            beta2=0.99,
            eps_adam=1e-8,
            input_path="input.txt",
            final_loss=2.5,
            samples=["ann"],
            run_timing=timing,
        )
        text = "\n".join(lines)
        parsed = parse_run_report_text(text)
        self.assertIsNotNone(parsed.run_timing)
        assert parsed.run_timing is not None
        self.assertEqual(parsed.run_timing.started_utc, timing.started_utc)
        self.assertEqual(parsed.run_timing.ended_local, timing.ended_local)

    def test_parse_filename_local_timestamp(self) -> None:
        ts = parse_filename_local_timestamp(
            "output_L1_E16_H4_B16_S1000_T0p5_seed42_20260424_152649.txt"
        )
        self.assertEqual(ts, "2026-04-24T15:26:49 (local, from filename)")

    def test_filename_hint_when_no_timing_block(self) -> None:
        text = """microGPT run report
===================
--- Config (this run) ---
N_LAYER=1
N_EMBD=16
N_HEAD=4
BLOCK_SIZE=16
NUM_STEPS=1
TEMPERATURE=0.5
SEED=42
LEARNING_RATE=0.01
BETA1=0.85
BETA2=0.99
EPS_ADAM=1e-08
INPUT_PATH=input.txt

Final loss (last training step): 2.500000

--- Inference samples ---
Sample  1: ann
"""
        parsed = parse_run_report_text(
            text,
            report_filename="output_L1_E16_H4_B16_S1000_T0p5_seed42_20260424_152649.txt",
        )
        self.assertIsNone(parsed.run_timing)
        self.assertIn("2026-04-24T15:26:49", parsed.filename_timestamp_hint or "")

    def test_write_read_sweep_timing(self) -> None:
        start = datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=3)
        timing = build_sweep_timing(start, end, run_count=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep_timing.txt"
            write_sweep_timing(path, timing)
            parsed = read_sweep_timing(path)
            text = path.read_text(encoding="utf-8")
        assert parsed is not None
        self.assertEqual(parsed.run_count, 4)
        self.assertAlmostEqual(parsed.duration_seconds, timing.duration_seconds)
        round_trip = parse_sweep_timing(text)
        assert round_trip is not None
        self.assertEqual(round_trip.started_utc, timing.started_utc)

    def test_format_duration_seconds(self) -> None:
        self.assertEqual(format_duration_seconds(45.2), "45.2s")
        self.assertEqual(format_duration_seconds(83), "1m 23s")
        self.assertEqual(format_duration_seconds(3661), "1h 1m")

    def test_format_progress_elapsed_eta(self) -> None:
        start = datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc)
        now = start + timedelta(seconds=100)
        line = format_progress_elapsed_eta(
            start, completed=10, total=100, now=now
        )
        self.assertIn("elapsed 1m 40s", line)
        self.assertIn("ETA 15m 0s", line)
        pending = format_progress_elapsed_eta(start, completed=0, total=4, now=now)
        self.assertIn("ETA —", pending)
        done = format_progress_elapsed_eta(start, completed=4, total=4, now=now)
        self.assertIn("ETA 0s", done)

    def test_print_carriage_progress_includes_eta(self) -> None:
        line = format_progress_elapsed_eta(
            datetime(2026, 5, 23, 10, 0, 0, tzinfo=timezone.utc),
            completed=250,
            total=2000,
            now=datetime(2026, 5, 23, 10, 0, 31, tzinfo=timezone.utc),
        )
        msg = f"Step  250 / 2000 | Loss 2.1913 | Avg-100 2.5245 | {line}"
        self.assertIn("ETA", msg)
        padded = msg.ljust(CARRIAGE_PROGRESS_WIDTH)
        self.assertGreaterEqual(len(padded), len(msg))
        self.assertIn("ETA", padded)


if __name__ == "__main__":
    unittest.main()
