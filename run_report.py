"""
microGPT run report: on-disk text format, parsing, and narrative (stdlib only).

Single source of truth for ``output_*.txt`` content consumed by
``microgpt_updated.py``, ``compare_run_reports.py``, and ``annotate_run_reports.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

# Keys required to build the narrative in ``format_run_narrative_lines`` / annotate.
NARRATIVE_REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "N_LAYER",
        "N_EMBD",
        "N_HEAD",
        "HEAD_DIM",
        "BLOCK_SIZE",
        "NUM_STEPS",
        "TEMPERATURE",
        "SEED",
        "INPUT_PATH",
    }
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


NARRATIVE_SECTION_HEADER = "--- What this run is ---"


def format_run_narrative_lines(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
    head_dim: int,
    block_size: int,
    num_steps: int,
    temperature: float,
    seed: int,
    input_path: str,
    final_loss: float,
    num_samples: int,
) -> list[str]:
    """Short human-readable story of inputs, training, and report outputs.

    Kept in one place so :func:`build_run_report_lines` and
    ``annotate_run_reports.py`` stay aligned for past and future runs.
    """
    in_name = Path(input_path).name if input_path else "(unknown file)"
    return [
        NARRATIVE_SECTION_HEADER,
        "",
        "Input: Training text is read from a plain-text file, one short document per "
        f"line (here: `{in_name}`). Characters become vocabulary tokens, plus a "
        "special beginning-of-sequence (BOS) token so the model learns where a line "
        "starts and when to stop. Context length is at most "
        f"{block_size} token positions (including BOS).",
        "",
        "Training: A small decoder-only transformer predicts the next character at "
        "each position (cross-entropy). This run uses "
        f"{n_layer} layer(s), width {n_embd}, {n_head} attention head(s) with "
        f"head width {head_dim}, trained for {num_steps} optimizer steps. "
        f"Init and sampling use seed {seed}; Adam with linear learning-rate decay is "
        "used during training (see config block below).",
        "",
        f"Output: The last-step training loss is {final_loss:.6f} (log-probability of "
        "the held-in-graph targets; lower is better on this training objective). "
        f"Below, {num_samples} lines are *generated* strings sampled from the learned "
        f"distribution (temperature {temperature}): they are not copies from the file; "
        "they show what the model has generalized.",
        "",
    ]


def run_parameter_glossary_lines() -> list[str]:
    """Human-readable meanings for report fields and output filename tokens."""
    return [
        "--- Parameter glossary ---",
        "Architecture / data:",
        "  N_LAYER   — Transformer depth (stacked attention+MLP blocks).",
        "  N_EMBD    — Hidden width; each token position is a vector this long.",
        "  N_HEAD    — Attention heads; splits N_EMBD into parallel subspaces.",
        "  HEAD_DIM  — N_EMBD // N_HEAD; per-head key/query/value width (must divide evenly).",
        "  BLOCK_SIZE — Max context length (positions 0..BLOCK_SIZE-1); must fit longest line + BOS.",
        "Training / optimisation:",
        "  NUM_STEPS — One training step = one document forward-backward + Adam update.",
        "  LEARNING_RATE — Base Adam step size (scaled by linear decay to 0 over the run).",
        "  BETA1, BETA2 — Adam momentum and variance decay.",
        "  EPS_ADAM — Numerical stability in Adam denominator.",
        "  SEED — RNG seed (init + shuffling + sampling) for reproducibility.",
        "Inference:",
        "  TEMPERATURE — Logits divided by this before softmax; lower = sharper samples.",
        "Data:",
        "  INPUT_PATH — Training text file (one document per line; BOS wraps each line).",
        "",
        "Filename tokens (output_L…_E…_H…_D…_B…_S…_T…_seed…_YYYYMMDD_HHMMSS.txt):",
        "  L = N_LAYER, E = N_EMBD, H = N_HEAD, D = HEAD_DIM, B = BLOCK_SIZE,",
        "  S = NUM_STEPS, T = TEMPERATURE (decimal point written as 'p', e.g. 0p5).",
        "  YYYYMMDD_HHMMSS = local wall-clock time when the report path is built.",
        "",
    ]


def format_run_output_path_for_params(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
    head_dim: int,
    block_size: int,
    num_steps: int,
    temperature: float,
    seed: int,
    prefix: str = "output",
    directory: str | Path = ".",
) -> Path:
    """Build a filesystem-safe path from hyperparameters and current local time.

    Example: ``output_L1_E16_H4_D4_B16_S1000_T0p5_seed42_20260422_153045.txt``
    (temperature dots become ``p``; trailing ``YYYYMMDD_HHMMSS`` is local wall-clock
    time when this function runs).
    """
    t_token = f"{temperature:g}".replace("-", "m").replace(".", "p")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"{prefix}_L{n_layer}_E{n_embd}_H{n_head}_D{head_dim}_B{block_size}"
        f"_S{num_steps}_T{t_token}_seed{seed}_{run_ts}"
    )
    return Path(directory) / f"{stem}.txt"


def build_run_report_lines(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
    head_dim: int,
    block_size: int,
    num_steps: int,
    temperature: float,
    seed: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps_adam: float,
    input_path: str,
    final_loss: float,
    samples: list[str],
    experiment_suite_lines: list[str] | None = None,
) -> list[str]:
    """Assemble the full run report as lines (no trailing newline on last line — caller joins)."""
    extra = experiment_suite_lines if experiment_suite_lines is not None else []
    lines = [
        "microGPT run report",
        "===================",
        *format_run_narrative_lines(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            head_dim=head_dim,
            block_size=block_size,
            num_steps=num_steps,
            temperature=temperature,
            seed=seed,
            input_path=input_path,
            final_loss=final_loss,
            num_samples=len(samples),
        ),
        *extra,
        "--- Config (this run) ---",
        f"N_LAYER={n_layer}",
        f"N_EMBD={n_embd}",
        f"N_HEAD={n_head}",
        f"HEAD_DIM={head_dim}",
        f"BLOCK_SIZE={block_size}",
        f"NUM_STEPS={num_steps}",
        f"TEMPERATURE={temperature}",
        f"SEED={seed}",
        f"LEARNING_RATE={learning_rate}",
        f"BETA1={beta1}",
        f"BETA2={beta2}",
        f"EPS_ADAM={eps_adam}",
        f"INPUT_PATH={input_path}",
        "",
        f"Final loss (last training step): {final_loss:.6f}",
        "",
        "--- Inference samples ---",
    ]
    for i, name in enumerate(samples, start=1):
        lines.append(f"Sample {i:2d}: {name}")
    lines.append("")
    lines.extend(run_parameter_glossary_lines())
    return lines
