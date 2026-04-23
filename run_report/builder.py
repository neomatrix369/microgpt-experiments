"""Assemble full run report text from structured fields."""

from __future__ import annotations

from .narrative import format_run_narrative_lines, run_parameter_glossary_lines


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
