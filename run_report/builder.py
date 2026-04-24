"""Assemble full run report text from structured fields."""

from __future__ import annotations

from .narrative import format_run_narrative_lines, run_parameter_glossary_lines


def build_run_report_lines(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
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
    loss_history: list[float] | None = None,
    experiment_suite_lines: list[str] | None = None,
    char_dist_score: float | None = None,
    quality_metrics: dict[str, float] | None = None,
    semantic_quality: dict[str, object] | None = None,
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
        f"# Per-head width (derived from N_EMBD, N_HEAD; not a separate run flag): "
        f"HEAD_DIM = N_EMBD // N_HEAD = {n_embd // n_head}",
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
    ]
    if char_dist_score is not None:
        lines.append("--- Sample quality (character-level) ---")
        lines.append(f"CHAR_DIST_SIMILARITY={char_dist_score:.6f}")
        if quality_metrics is not None:
            lines.append(f"AVG_SAMPLE_LENGTH={quality_metrics['avg_sample_length']:.6f}")
            lines.append(f"LENGTH_SIMILARITY={quality_metrics['length_similarity']:.6f}")
        lines.append("")
    if semantic_quality is not None:
        lines.append("--- Semantic quality (three-tier) ---")
        lines.append(f"TIER1_REAL_COUNT={int(semantic_quality['tier1_real_count'])}")
        lines.append(f"TIER1_REAL_RATIO={float(semantic_quality['tier1_real_ratio']):.6f}")
        lines.append(f"TIER2_PLAUSIBLE_COUNT={int(semantic_quality['tier2_plausible_count'])}")
        lines.append(
            f"TIER2_PLAUSIBLE_RATIO={float(semantic_quality['tier2_plausible_ratio']):.6f}"
        )
        lines.append(f"TIER2_AVG_SCORE={float(semantic_quality['tier2_avg_score']):.6f}")
        lines.append(f"TIER3_NONSENSE_COUNT={int(semantic_quality['tier3_nonsense_count'])}")
        lines.append(
            f"TIER3_NONSENSE_RATIO={float(semantic_quality['tier3_nonsense_ratio']):.6f}"
        )
        lines.append(
            f"OVERALL_QUALITY_SCORE={float(semantic_quality['overall_quality_score']):.6f}"
        )
        lines.append("")
        ex1 = semantic_quality.get("tier1_examples") or []
        ex2 = semantic_quality.get("tier2_examples") or []
        ex3 = semantic_quality.get("tier3_examples") or []
        if ex1:
            lines.append("# Tier 1 Examples (Real):")
            lines.append(f"#   {', '.join(str(x) for x in ex1)}")
        if ex2:
            lines.append("# Tier 2 Examples (Plausible):")
            lines.append(f"#   {', '.join(str(x) for x in ex2)}")
        if ex3:
            lines.append("# Tier 3 Examples (Nonsense):")
            lines.append(f"#   {', '.join(str(x) for x in ex3)}")
        lines.append("")
    if loss_history:
        lines.append("--- Loss history (CSV: step,loss) ---")
        for i, v in enumerate(loss_history):
            lines.append(f"{i},{v:.6f}")
        lines.append("")
    lines.append("--- Inference samples ---")
    for i, name in enumerate(samples, start=1):
        lines.append(f"Sample {i:2d}: {name}")
    lines.append("")
    lines.extend(run_parameter_glossary_lines())
    return lines
