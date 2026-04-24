"""run_report.paths helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from run_report.paths import DEFAULT_RUN_REPORT_DIR, run_reports_dir


class RunReportsDirTest(unittest.TestCase):
    def test_joins_repo_root(self) -> None:
        root = Path("/fake/repo")
        self.assertEqual(
            run_reports_dir(root),
            root / DEFAULT_RUN_REPORT_DIR,
        )


if __name__ == "__main__":
    unittest.main()
