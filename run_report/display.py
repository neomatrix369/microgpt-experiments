"""Shared formatting for experiment config keys in CLI and HTML output."""

from __future__ import annotations

from .parse import experiment_cfg_calculated_caption


def format_cfg_value(val: object) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def config_print_label(key: str) -> str:
    cap = experiment_cfg_calculated_caption(key)
    return f"{key} ({cap})" if cap else key


def format_config_line(key: str, val: object) -> str:
    cap = experiment_cfg_calculated_caption(key)
    suffix = f"  — {cap}" if cap else ""
    return f"{key}={format_cfg_value(val)}{suffix}"


def format_shared_config_line(key: str, val: object) -> str:
    """Config line for HTML/text shared blocks (includes calculated caption)."""
    line = f"{key}={format_cfg_value(val)}"
    cap = experiment_cfg_calculated_caption(key)
    if cap:
        line += f"  — {cap}"
    return line
