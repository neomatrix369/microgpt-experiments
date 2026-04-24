"""Human-readable narrative and glossary for run reports."""

from __future__ import annotations

from pathlib import Path

from .paths import DEFAULT_RUN_REPORT_DIR

# Keys required to build the narrative in ``format_run_narrative_lines`` / annotate.
NARRATIVE_REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "N_LAYER",
        "N_EMBD",
        "N_HEAD",
        "BLOCK_SIZE",
        "NUM_STEPS",
        "TEMPERATURE",
        "SEED",
        "INPUT_PATH",
    }
)

NARRATIVE_SECTION_HEADER = "--- What this run is ---"


def format_run_narrative_lines(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
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
    per_head = n_embd // n_head if n_head else 0
    head_word = "head" if n_head == 1 else "heads"
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
        "each position (cross-entropy). Architecture: "
        f"{n_layer} layer(s), hidden width N_EMBD={n_embd}, N_HEAD={n_head} "
        f"({n_embd} split across {n_head} {head_word} → {per_head} per head), trained for "
        f"{num_steps} optimizer steps. "
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
        "  N_HEAD    — Number of attention heads; chosen with N_EMBD (per-head width is N_EMBD//N_HEAD).",
        "  HEAD_DIM  — Calculated from N_EMBD and N_HEAD (N_EMBD // N_HEAD); stored as HEAD_DIM=;",
        "            comparison tools annotate it as a derived field, not a separate sweep knob.",
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
        "Filename tokens (output_L…_E…_H…_B…_S…_T…_seed…_YYYYMMDD_HHMMSS.txt):",
        "  L = N_LAYER, E = N_EMBD, H = N_HEAD, B = BLOCK_SIZE,",
        "  S = NUM_STEPS, T = TEMPERATURE (decimal point written as 'p', e.g. 0p5).",
        "  YYYYMMDD_HHMMSS = local wall-clock time when the report path is built.",
        f"  Default folder: <microgpt repo>/{DEFAULT_RUN_REPORT_DIR.as_posix()}/ "
        "(see run_report.paths.run_reports_dir).",
        "",
    ]
