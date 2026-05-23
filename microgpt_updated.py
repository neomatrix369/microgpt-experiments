"""
Train and run inference for a GPT in pure, dependency-free Python.

Entry script: hyperparameters, training loop, generation, and run-report wiring.
The scalar autograd ``Value``, transformer forward, and data helpers live in
``mgpt``; ``output_*.txt`` format and parsing live in ``run_report``.

Hyperparameters: edit module-level constants and/or pass CLI flags (see
``python microgpt_updated.py --help``). Primary width/attention controls are
``N_EMBD`` and ``N_HEAD``; ``HEAD_DIM`` is ``N_EMBD // N_HEAD`` and appears in saved
reports immediately after those two as ``HEAD_DIM=`` (plus a ``#`` note). Run reports default to
``run_reports_dir(Path(__file__).resolve().parent)`` (the ``outputs/`` folder next to this
script). User-facing overview: ``README.md`` (sections *How to use microgpt_updated.py*
and *Run experiments examples*).

@karpathy
https://karpathy.github.io/2026/02/12/microgpt/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mgpt.experiment import (
    RunConfig,
    generate,
    run_experiment,
    save_run_report,
    train,
)
from run_report import (
    DEFAULT_RUN_REPORT_DIR,
    format_run_output_path_for_params,
    run_reports_dir,
)

_MICROGPT_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_RUN_REPORTS_DIR = run_reports_dir(_MICROGPT_REPO_ROOT)

# Hyperparameters
# ====================

# Depth of the transformer (number of layers). 1 is enough for
# character-level name generation; deeper nets help for harder tasks but
# cost linearly more here because our scalar autograd is O(nodes) per
# backward pass and each layer adds a fixed number of nodes.
# tunable parameter
# Unit: layers (count).
N_LAYER = 1

# Width of the network (embedding dimension). Each token and position is
# represented as a vector of this length. 16 is tiny but trains fast in
# pure Python and is enough to learn spelling patterns in short names.
# tunable parameter
# Unit: dimensions.
N_EMBD = 16

# Number of attention heads. Multi-head attention lets the model attend to
# different positional relationships in parallel. 4 heads of dimension 4
# (16 / 4) is a reasonable split for this embedding size.
# tunable parameter (fixed by the embedding size and number of heads)
# Unit: heads (count).
N_HEAD = 4

# Derived dimension of each attention head (N_EMBD // N_HEAD).
# Unit: dimensions.
HEAD_DIM = N_EMBD // N_HEAD

# Maximum context length of the attention window. The longest name in the
# dataset is 15 characters, so 16 covers every example with room for BOS.
# tunable parameter (fixed for this dataset)
# Unit: tokens (positions).
BLOCK_SIZE = 16

# Initial learning rate for Adam. 0.01 is on the high side for larger
# models but works well here because the model is small and we apply
# linear decay, so the effective rate drops to zero by the final step.
# Unit: dimensionless (scalar step scale; multiplied by linear decay).
LEARNING_RATE = 0.01

# Adam first-moment decay (beta1). Controls how much the optimiser trusts
# the current gradient vs the running average. Standard default is 0.9;
# 0.85 forgets faster, which helps on a tiny noisy dataset where stale
# momentum would overshoot.
# Unit: dimensionless (decay factor in (0, 1)).
BETA1 = 0.85

# Adam second-moment decay (beta2). Controls the running average of
# squared gradients used to scale the step size per-parameter. 0.99 is
# slightly more aggressive than the typical 0.999, giving faster
# adaptation at the cost of noisier variance estimates (fine here).
# Unit: dimensionless (decay factor in (0, 1)).
BETA2 = 0.99

# Adam epsilon. Added to the denominator to prevent division by zero when
# a parameter's gradient history is near-zero. 1e-8 is the standard
# default.
# Unit: dimensionless.
EPS_ADAM = 1e-8

# Total number of training steps. Each step processes one document. 1000
# is enough for convergence on this dataset (32k names, vocab of 27
# characters).
# Unit: steps (one forward-backward + Adam update per step).
NUM_STEPS = 1000

# Sampling temperature in (0, 1]. Lower values sharpen the distribution
# (more conservative, common names), higher values flatten it (more
# creative, weirder names). 0.5 is a good middle ground.
# Unit: dimensionless (logits divided by this before softmax).
TEMPERATURE = 0.5

SEED = 42  # Unit: dimensionless (integer RNG seed for init and sampling).
NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
INPUT_PATH = "input.txt"

# Optional labels for variant sweeps and run reports. Set ``EXPERIMENT_SUITE_INDEX``
# to the 1-based position of this run; set ``EXPERIMENT_SUITE_TOTAL`` when you
# know how many runs the suite will contain (both None omits the suite line).
EXPERIMENT_SUITE_INDEX: int | None = None
EXPERIMENT_SUITE_TOTAL: int | None = None
# Optional one-line context for a sweep (printed under ``--- Experiment suite ---``).
EXPERIMENT_SUITE_NOTE: str | None = None


def _module_default_config() -> RunConfig:
    return RunConfig(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        block_size=BLOCK_SIZE,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        eps_adam=EPS_ADAM,
        input_path=INPUT_PATH,
        names_url=NAMES_URL,
        suite_index=EXPERIMENT_SUITE_INDEX,
        suite_total=EXPERIMENT_SUITE_TOTAL,
        suite_note=EXPERIMENT_SUITE_NOTE,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI overrides for hyperparameters; omitted flags keep module defaults."""
    p = argparse.ArgumentParser(
        description=(
            "Train the tiny GPT, print samples and quality metrics, "
            "and save output_*.txt (same as running with no arguments "
            "using the constants at the top of this file)."
        )
    )
    g = p.add_argument_group("hyperparameters (defaults match this file)")
    g.add_argument(
        "--n-layer", type=int, default=N_LAYER, metavar="N",
        help="transformer depth (unit: layers)",
    )
    g.add_argument(
        "--n-embd", type=int, default=N_EMBD, metavar="N",
        help="hidden / embedding width (unit: dimensions)",
    )
    g.add_argument(
        "--n-head", type=int, default=N_HEAD, metavar="N",
        help="attention head count (unit: heads; N_EMBD must be divisible)",
    )
    g.add_argument(
        "--block-size", type=int, default=BLOCK_SIZE, metavar="N",
        help="max context length (unit: tokens / positions)",
    )
    g.add_argument(
        "--num-steps", type=int, default=NUM_STEPS, metavar="N",
        help="training optimizer steps (unit: steps)",
    )
    g.add_argument(
        "--temperature", type=float, default=TEMPERATURE, metavar="T",
        help="sampling temperature (unit: dimensionless)",
    )
    g.add_argument(
        "--seed", type=int, default=SEED, metavar="N",
        help="RNG seed (unit: dimensionless integer)",
    )
    g.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, metavar="LR",
        help="base Adam learning rate (unit: dimensionless)",
    )
    g.add_argument(
        "--beta1", type=float, default=BETA1,
        help="Adam beta1 (unit: dimensionless decay factor)",
    )
    g.add_argument(
        "--beta2", type=float, default=BETA2,
        help="Adam beta2 (unit: dimensionless decay factor)",
    )
    g.add_argument(
        "--input",
        default=INPUT_PATH,
        metavar="PATH",
        help="Training text file (one document per line); default from INPUT_PATH.",
    )
    s = p.add_argument_group(
        "experiment suite (optional; only applied if you pass the flag)"
    )
    s.add_argument("--suite-index", type=int, default=argparse.SUPPRESS, metavar="N")
    s.add_argument("--suite-total", type=int, default=argparse.SUPPRESS, metavar="N")
    s.add_argument("--suite-note", default=argparse.SUPPRESS, metavar="TEXT")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for the run report .txt "
            f"(default: <microgpt repo>/{DEFAULT_RUN_REPORT_DIR}, "
            "same folder as this script)."
        ),
    )
    return p


def _apply_parsed_args(args: argparse.Namespace) -> None:
    """Copy argparse results into module-level hyperparameters (for readers of this file)."""
    global N_LAYER, N_EMBD, N_HEAD, HEAD_DIM, BLOCK_SIZE, NUM_STEPS, TEMPERATURE, SEED
    global LEARNING_RATE, BETA1, BETA2, INPUT_PATH
    global EXPERIMENT_SUITE_INDEX, EXPERIMENT_SUITE_TOTAL, EXPERIMENT_SUITE_NOTE

    N_LAYER = args.n_layer
    N_EMBD = args.n_embd
    N_HEAD = args.n_head
    HEAD_DIM = N_EMBD // N_HEAD
    BLOCK_SIZE = args.block_size
    NUM_STEPS = args.num_steps
    TEMPERATURE = args.temperature
    SEED = args.seed
    LEARNING_RATE = args.learning_rate
    BETA1 = args.beta1
    BETA2 = args.beta2
    INPUT_PATH = args.input

    if hasattr(args, "suite_index"):
        EXPERIMENT_SUITE_INDEX = args.suite_index
    if hasattr(args, "suite_total"):
        EXPERIMENT_SUITE_TOTAL = args.suite_total
    if hasattr(args, "suite_note"):
        EXPERIMENT_SUITE_NOTE = args.suite_note


def _validate_hyperparameters(parser: argparse.ArgumentParser) -> None:
    try:
        _module_default_config().validate()
    except ValueError as exc:
        parser.error(str(exc))


def format_run_output_path(
    *,
    prefix: str = "output",
    directory: str | Path = _DEFAULT_RUN_REPORTS_DIR,
) -> Path:
    """Build a filesystem-safe path from the current module hyperparameters."""
    return format_run_output_path_for_params(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        block_size=BLOCK_SIZE,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        seed=SEED,
        prefix=prefix,
        directory=directory,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    _apply_parsed_args(args)
    _validate_hyperparameters(parser)

    config = RunConfig.from_argparse_namespace(args, defaults=_module_default_config())
    report_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else _DEFAULT_RUN_REPORTS_DIR
    )
    run_experiment(config, output_dir=report_dir, save_report=True)


if __name__ == "__main__":
    main()
