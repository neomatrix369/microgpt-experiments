"""Parse saved ``output_*.txt`` run reports."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CFG_INT = frozenset(
    {"N_LAYER", "N_EMBD", "N_HEAD", "HEAD_DIM", "BLOCK_SIZE", "NUM_STEPS", "SEED"}
)
_CFG_FLOAT = frozenset(
    {"TEMPERATURE", "LEARNING_RATE", "BETA1", "BETA2", "EPS_ADAM"}
)
_CFG_STR = frozenset({"INPUT_PATH"})

# Order used when printing a single shared config (both runs agree) — and display.
_CFG_DISPLAY_ORDER: tuple[str, ...] = (
    "N_LAYER",
    "N_EMBD",
    "N_HEAD",
    "HEAD_DIM",
    "BLOCK_SIZE",
    "NUM_STEPS",
    "TEMPERATURE",
    "SEED",
    "LEARNING_RATE",
    "BETA1",
    "BETA2",
    "EPS_ADAM",
    "INPUT_PATH",
)


@dataclass(frozen=True)
class ParsedRunReport:
    config: dict[str, int | float | str]
    final_loss: float
    samples: list[str]


def cfg_keys_in_display_order(keys: set[str]) -> list[str]:
    ordered = [k for k in _CFG_DISPLAY_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in _CFG_DISPLAY_ORDER)
    return ordered + rest


def parse_run_report_text(text: str) -> ParsedRunReport:
    """Parse a saved run report: config key=value lines, final loss, inference samples."""
    cfg: dict[str, int | float | str] = {}
    for line in text.splitlines():
        if not line or line.startswith("---"):
            continue
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if key in _CFG_INT:
            try:
                cfg[key] = int(rest.strip())
            except ValueError:
                pass
        elif key in _CFG_FLOAT:
            try:
                cfg[key] = float(rest.strip())
            except ValueError:
                pass
        elif key in _CFG_STR:
            cfg[key] = rest.strip()

    m_loss = re.search(
        r"^Final loss \(last training step\): ([0-9.eE+-]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not m_loss:
        raise ValueError("missing final loss line")
    final_loss = float(m_loss.group(1))

    samples: list[str] = []
    in_samples = False
    sample_re = re.compile(r"^Sample\s+\d+:\s*(.*)$")
    for line in text.splitlines():
        if line.strip() == "--- Inference samples ---":
            in_samples = True
            continue
        if in_samples:
            if line.startswith("---"):
                break
            m = sample_re.match(line.rstrip())
            if m:
                samples.append(m.group(1))

    return ParsedRunReport(config=cfg, final_loss=final_loss, samples=samples)
