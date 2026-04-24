# M2+ Semantic quality (execution log)

Three-tier **heuristic** sample evaluation (training-corpus membership, plausibility score, nonsense flags) plus character-level distribution similarity. Stdlib-only; lives in `mgpt/evaluation.py`.

## Slices (completed)

| Slice | Outcome |
|-------|---------|
| **1 — `mgpt/evaluation.py`** | `char_distribution_similarity`, `evaluate_sample_quality`, `is_pronounceable`, `score_plausibility`, `classify_plausible_words`, `is_nonsense`, `count_nonsense_words`, `evaluate_semantic_quality`. Max consonant run **4** (English-friendly); `overall_quality_score` clamped to `[0, 1]`. |
| **2 — `run_report`** | `build_run_report_lines(..., char_dist_score=, quality_metrics=, semantic_quality=)`. `ParsedRunReport` extended; `parse_run_report_text` reads `--- Sample quality ---` and `--- Semantic quality ---` blocks and example comment lines. |
| **3 — `microgpt_updated.py`** | After `generate()`, prints the SAMPLE QUALITY block and passes metrics into `save_run_report()`. |
| **4 — `experiments/report_generator.py`** | `python experiments/report_generator.py [output_*.txt ...] -o comparison_report.html` builds a comparison table and tier bar rows. |
| **5 — Tests** | `tests/test_evaluation.py` (pronounce/nonsense, tier extremes, score bounds, builder/parse round-trip). |

## Commands

```bash
python microgpt_updated.py
python -m pytest tests/test_evaluation.py tests/test_text_loss_plot.py -q
python experiments/report_generator.py -o comparison_report.html
# (script adds repo root to sys.path so it works from any cwd)
```

## Notes

- **Reuse**: `compute_sample_quality_metrics` + `format_sample_quality_console_lines` centralize what `microgpt_updated.py` prints and saves; semantic evaluation uses one pass over samples with cached corpus bigrams / average length. HTML reports prefer `HEAD_DIM` from the parsed config when present.
- **Tier overlap**: Tier 1 (real), Tier 2 (plausible among **non-real**), and Tier 3 (nonsense on **all** samples) are not mutually exclusive by construction; `distribution_sum` in `evaluate_semantic_quality` is a sanity hint only.
- **Hypothesis testing (H×D sweeps)**: Compare `OVERALL_QUALITY_SCORE` and tier ratios across `output_*.txt` reports or the HTML summary.
