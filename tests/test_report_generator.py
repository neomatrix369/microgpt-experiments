"""HTML report generator: legacy vs modern (semantic quality) reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.report_generator import generate_html_report

_MODERN_TAIL = """
--- Semantic quality (three-tier) ---
TIER1_REAL_COUNT=1
TIER1_REAL_RATIO=0.05
TIER2_PLAUSIBLE_COUNT=6
TIER2_PLAUSIBLE_RATIO=0.3
TIER2_AVG_SCORE=0.44
TIER3_NONSENSE_COUNT=13
TIER3_NONSENSE_RATIO=0.65
OVERALL_QUALITY_SCORE=0.182

--- Inference samples ---
Sample  1: alex
"""

_LEGACY_TAIL = """
--- Inference samples ---
Sample  1: zzz
"""


def _report_body(*, modern_semantic: bool) -> str:
    base = """N_LAYER=1
N_EMBD=16
N_HEAD=4
HEAD_DIM=4
BLOCK_SIZE=16
NUM_STEPS=1000
TEMPERATURE=0.5
SEED=42
Final loss (last training step): 2.0000
"""
    return base + (_MODERN_TAIL if modern_semantic else _LEGACY_TAIL)


class ReportGeneratorLegacyTest(unittest.TestCase):
    def test_collapses_legacy_into_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            leg = tdir / "legacy.txt"
            mod = tdir / "modern.txt"
            out = tdir / "out.html"
            leg.write_text(_report_body(modern_semantic=False), encoding="utf-8")
            mod.write_text(_report_body(modern_semantic=True), encoding="utf-8")
            generate_html_report([leg, mod], out)
            html = out.read_text(encoding="utf-8")
        self.assertIn("legacy-summary", html)
        self.assertIn("Legacy experiments (1):", html)
        self.assertIn("legacy.txt", html)
        self.assertIn("alex", html)
        # Header row + one modern data row + one legacy summary row
        self.assertEqual(html.count("<tr"), 3)

    def test_legacy_only_shows_message_in_quality_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            leg = tdir / "only_legacy.txt"
            out = tdir / "out.html"
            leg.write_text(_report_body(modern_semantic=False), encoding="utf-8")
            generate_html_report([leg], out)
            html = out.read_text(encoding="utf-8")
        self.assertIn("legacy-summary", html)
        self.assertIn("No semantic quality data", html)


if __name__ == "__main__":
    unittest.main()
