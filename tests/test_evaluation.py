"""Tests for three-tier semantic quality and character-level sample metrics."""

from __future__ import annotations

import unittest

from mgpt.evaluation import (
    char_distribution_similarity,
    compute_sample_quality_metrics,
    evaluate_sample_quality,
    evaluate_semantic_quality,
    format_sample_quality_console_lines,
    is_nonsense,
    is_pronounceable,
)
from run_report.builder import build_run_report_lines
from run_report.parse import parse_run_report_text


class TestPronounceableAndNonsense(unittest.TestCase):
    def test_alice_pronounceable(self) -> None:
        self.assertTrue(is_pronounceable("alice"))

    def test_zzxqwp_not_pronounceable(self) -> None:
        self.assertFalse(is_pronounceable("zzxqwp"))

    def test_aaaa_nonsense(self) -> None:
        self.assertTrue(is_nonsense("aaaa"))


class TestSemanticTiers(unittest.TestCase):
    def test_all_real_tier1_full(self) -> None:
        corpus = ["alice", "bob", "carl"]
        samples = ["alice", "bob", "carl"]
        r = evaluate_semantic_quality(samples, corpus)
        self.assertEqual(r["tier1_real_count"], 3)
        self.assertEqual(r["tier1_real_ratio"], 1.0)

    def test_all_nonsense_tier3_full(self) -> None:
        samples = ["aaaa"] * 20
        corpus = ["alice", "bob"]
        r = evaluate_semantic_quality(samples, corpus)
        self.assertEqual(r["tier3_nonsense_count"], 20)
        self.assertEqual(r["tier3_nonsense_ratio"], 1.0)

    def test_overall_quality_bounded(self) -> None:
        corpus = ["alice", "bob", "carl", "dave"]
        for _ in range(5):
            r = evaluate_semantic_quality(
                ["alice", "xyzqw", "m", "aeiou", "bbbb"],
                corpus,
            )
            self.assertGreaterEqual(r["overall_quality_score"], 0.0)
            self.assertLessEqual(r["overall_quality_score"], 1.0)


class TestComputeBundle(unittest.TestCase):
    def test_compute_matches_individual_evaluators(self) -> None:
        corpus = ["alice", "bob"]
        samples = ["alice", "zzz"]
        a, b, c = compute_sample_quality_metrics(samples, corpus)
        self.assertEqual(a, char_distribution_similarity(samples, corpus))
        self.assertEqual(b, evaluate_sample_quality(samples, corpus))
        self.assertEqual(c, evaluate_semantic_quality(samples, corpus))

    def test_format_lines_non_empty(self) -> None:
        _, qm, sq = compute_sample_quality_metrics(["a", "bb"], ["alice"])
        lines = format_sample_quality_console_lines(0.5, qm, sq, n_samples=2)
        self.assertTrue(any("SAMPLE QUALITY" in ln for ln in lines))


class TestCharMetrics(unittest.TestCase):
    def test_char_distribution_identical(self) -> None:
        docs = ["abc", "def"]
        samples = ["abc", "def"]
        s = char_distribution_similarity(samples, docs)
        self.assertAlmostEqual(s, 1.0, places=5)

    def test_sample_quality_length(self) -> None:
        docs = ["ab", "cd", "ef"]
        samples = ["a", "bbb"]
        q = evaluate_sample_quality(samples, docs)
        self.assertGreater(q["avg_sample_length"], 0.0)


class TestReportSemanticRoundTrip(unittest.TestCase):
    def test_parse_semantic_block(self) -> None:
        sem = {
            "tier1_real_count": 2,
            "tier1_real_ratio": 0.1,
            "tier2_plausible_count": 10,
            "tier2_plausible_ratio": 0.5,
            "tier2_avg_score": 0.72,
            "tier3_nonsense_count": 8,
            "tier3_nonsense_ratio": 0.4,
            "overall_quality_score": 0.55,
            "tier1_examples": ["alice"],
            "tier2_examples": ["jalice"],
            "tier3_examples": ["qqq"],
        }
        lines = build_run_report_lines(
            n_layer=1,
            n_embd=16,
            n_head=4,
            head_dim=4,
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
            samples=["a"],
            loss_history=None,
            experiment_suite_lines=None,
            char_dist_score=0.8,
            quality_metrics={"avg_sample_length": 5.0, "length_similarity": 0.9},
            semantic_quality=sem,
        )
        text = "\n".join(lines)
        p = parse_run_report_text(text)
        self.assertAlmostEqual(p.char_dist_score or 0.0, 0.8)
        self.assertAlmostEqual(p.avg_sample_length or 0.0, 5.0)
        self.assertIsNotNone(p.semantic_quality)
        sq = p.semantic_quality
        assert sq is not None
        self.assertEqual(sq["tier1_examples"], ["alice"])
        self.assertAlmostEqual(float(sq["overall_quality_score"]), 0.55)


if __name__ == "__main__":
    unittest.main()
