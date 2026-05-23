"""CLI tests for compare_run_reports.py."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_COMPARE = _REPO / "compare_run_reports.py"
_EXAMPLES = _REPO / "example-experiments"
_REPORT_A = _EXAMPLES / "output_L1_E16_H4_B16_S1000_T0p5_seed42_20260424_152649.txt"
_REPORT_B = _EXAMPLES / "output_L1_E16_H1_B16_S1000_T0p5_seed42_20260424_152836.txt"


def _run_compare(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_COMPARE), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


class TestCompareRunReportsCli(unittest.TestCase):
    def test_given_identical_reports_when_compared_then_exit_zero(self) -> None:
        if not _REPORT_A.is_file():
            self.skipTest("example report A missing")
        result = _run_compare(str(_REPORT_A), str(_REPORT_A))
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_given_different_reports_when_compared_then_exit_one(self) -> None:
        if not _REPORT_A.is_file() or not _REPORT_B.is_file():
            self.skipTest("example reports missing")
        result = _run_compare(str(_REPORT_A), str(_REPORT_B))
        self.assertEqual(result.returncode, 1, msg=result.stderr)
        self.assertIn("HEAD_DIM", result.stdout)

    def test_given_missing_file_when_compared_then_exit_two(self) -> None:
        result = _run_compare("/no/such/file.txt", "/also/missing.txt")
        self.assertEqual(result.returncode, 2)

    def test_given_bad_args_when_compared_then_exit_two(self) -> None:
        result = _run_compare("only_one_arg.txt")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
