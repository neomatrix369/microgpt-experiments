"""CLI tests for annotate_run_reports.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_report.builder import build_run_report_lines

_REPO = Path(__file__).resolve().parent.parent
_ANNOTATE = _REPO / "annotate_run_reports.py"


def _run_annotate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_ANNOTATE), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def _minimal_report_without_narrative() -> str:
    lines = build_run_report_lines(
        n_layer=1,
        n_embd=16,
        n_head=4,
        block_size=16,
        num_steps=10,
        temperature=0.5,
        seed=42,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.99,
        eps_adam=1e-8,
        input_path="input.txt",
        final_loss=2.5,
        samples=["alice", "bob"],
        loss_history=None,
    )
    text = "\n".join(lines)
    marker = "--- What this run is ---"
    start = text.find(marker)
    if start < 0:
        return text
    end = text.find("\n--- Config (this run) ---", start)
    if end < 0:
        return text
    return text[:start] + text[end + 1 :]


class TestAnnotateRunReportsCli(unittest.TestCase):
    def test_given_report_without_narrative_when_annotated_then_inserts_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output_test.txt"
            path.write_text(_minimal_report_without_narrative(), encoding="utf-8")
            result = _run_annotate(str(path))
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            text = path.read_text(encoding="utf-8")
            self.assertIn("--- What this run is ---", text)

    def test_given_report_with_narrative_when_annotated_then_skips(self) -> None:
        example = (
            _REPO
            / "example-experiments"
            / "output_L1_E16_H4_B16_S1000_T0p5_seed42_20260424_152649.txt"
        )
        if not example.is_file():
            self.skipTest("example report missing")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output_copy.txt"
            path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            result = _run_annotate(str(path))
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("skip", result.stdout.lower())

    def test_given_unparseable_file_when_annotated_then_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text("not a valid report\n", encoding="utf-8")
            result = _run_annotate(str(path))
            self.assertEqual(result.returncode, 1, msg=result.stdout)


if __name__ == "__main__":
    unittest.main()
