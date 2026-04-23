"""
microGPT run report: on-disk text format, parsing, and narrative (stdlib only).

Single source of truth for ``output_*.txt`` content consumed by
``microgpt_updated.py``, ``compare_run_reports.py``, and ``annotate_run_reports.py``.
"""

from __future__ import annotations

from .builder import build_run_report_lines
from .narrative import (
    NARRATIVE_REQUIRED_CONFIG_KEYS,
    NARRATIVE_SECTION_HEADER,
    format_run_narrative_lines,
    run_parameter_glossary_lines,
)
from .parse import ParsedRunReport, cfg_keys_in_display_order, parse_run_report_text
from .paths import format_run_output_path_for_params
from .text_loss_plot import bin_mean, bin_stats, bin_step_ranges, loss_curve_comparison_lines

__all__ = [
    "ParsedRunReport",
    "bin_mean",
    "bin_stats",
    "bin_step_ranges",
    "build_run_report_lines",
    "cfg_keys_in_display_order",
    "format_run_narrative_lines",
    "format_run_output_path_for_params",
    "loss_curve_comparison_lines",
    "NARRATIVE_REQUIRED_CONFIG_KEYS",
    "NARRATIVE_SECTION_HEADER",
    "parse_run_report_text",
    "run_parameter_glossary_lines",
]
