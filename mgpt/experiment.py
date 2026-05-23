"""Programmatic training + evaluation API for microGPT runs."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from mgpt.data import build_tokeniser, load_dataset
from mgpt.evaluation import (
    compute_sample_quality_metrics,
    format_sample_quality_console_lines,
)
from mgpt.model import KVCache, StateDict, Tokeniser, gpt
from mgpt.ops import Vector, make_matrix, softmax
from mgpt.value import Value
from run_report import build_run_report_lines, format_run_output_path_for_params

DEFAULT_NAMES_URL = (
    "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
)

# Keys that must never appear in a sweep grid (derived or non-hyperparameter).
FORBIDDEN_GRID_KEYS = frozenset({"head_dim", "HEAD_DIM"})


@dataclass(frozen=True)
class RunConfig:
    """Hyperparameters for one training + generation run."""

    n_layer: int = 1
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16
    num_steps: int = 1000
    temperature: float = 0.5
    seed: int = 42
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    input_path: str = "input.txt"
    names_url: str = DEFAULT_NAMES_URL
    suite_index: int | None = None
    suite_total: int | None = None
    suite_note: str | None = None

    @classmethod
    def sweep_field_names(cls) -> frozenset[str]:
        """RunConfig fields allowed in sweep fixed/grid blocks."""
        return frozenset(
            f.name
            for f in fields(cls)
            if f.name not in {"suite_index", "suite_total", "suite_note", "names_url"}
        )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def validate(self) -> None:
        """Raise ValueError if hyperparameters are invalid."""
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")
        if self.n_embd < 1:
            raise ValueError("n_embd must be >= 1")
        if self.n_head < 1:
            raise ValueError("n_head must be >= 1")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        defaults: RunConfig | None = None,
    ) -> RunConfig:
        """Build from a dict (e.g. sweep fixed + grid combo). Unknown keys raise."""
        base = defaults or cls()
        allowed = {f.name for f in fields(cls)}
        unknown = set(mapping) - allowed - FORBIDDEN_GRID_KEYS
        if unknown:
            raise ValueError(f"unknown RunConfig keys: {sorted(unknown)}")
        if FORBIDDEN_GRID_KEYS & set(mapping):
            raise ValueError("head_dim is derived from n_embd and n_head; do not sweep it")
        kwargs = {f.name: getattr(base, f.name) for f in fields(cls)}
        for key, value in mapping.items():
            if key in allowed:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_argparse_namespace(cls, args: Any, *, defaults: RunConfig | None = None) -> RunConfig:
        """Build from parsed CLI namespace (microgpt_updated.py)."""
        base = defaults or cls()
        cfg = replace(
            base,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            block_size=args.block_size,
            num_steps=args.num_steps,
            temperature=args.temperature,
            seed=args.seed,
            learning_rate=args.learning_rate,
            beta1=args.beta1,
            beta2=args.beta2,
            input_path=args.input,
        )
        if hasattr(args, "suite_index"):
            cfg = replace(cfg, suite_index=args.suite_index)
        if hasattr(args, "suite_total"):
            cfg = replace(cfg, suite_total=args.suite_total)
        if hasattr(args, "suite_note"):
            cfg = replace(cfg, suite_note=args.suite_note)
        return cfg

    def to_cli_argv(self) -> list[str]:
        """Copy-pasteable CLI flags for this config (omits defaults-only suite fields)."""
        parts: list[str] = []
        defaults = RunConfig()
        flag_map = {
            "n_layer": "--n-layer",
            "n_embd": "--n-embd",
            "n_head": "--n-head",
            "block_size": "--block-size",
            "num_steps": "--num-steps",
            "temperature": "--temperature",
            "seed": "--seed",
            "learning_rate": "--learning-rate",
            "beta1": "--beta1",
            "beta2": "--beta2",
            "input_path": "--input",
        }
        for field_name, flag in flag_map.items():
            val = getattr(self, field_name)
            if val != getattr(defaults, field_name):
                parts.extend([flag, str(val)])
        return parts


@dataclass(frozen=True)
class ExperimentResult:
    """Outcome of one full train + generate + evaluate cycle."""

    config: RunConfig
    final_loss: float
    samples: list[str]
    loss_history: list[float]
    char_dist_score: float
    quality_metrics: dict[str, float]
    semantic_quality: dict[str, object]
    report_path: Path | None


def experiment_suite_lines(config: RunConfig) -> list[str]:
    if (
        config.suite_index is None
        and config.suite_total is None
        and config.suite_note is None
    ):
        return []
    lines = ["--- Experiment suite ---"]
    if config.suite_index is not None and config.suite_total is not None:
        lines.append(f"Experiment: {config.suite_index} / {config.suite_total}")
    elif config.suite_index is not None:
        lines.append(
            f"Experiment index: {config.suite_index} "
            "(set EXPERIMENT_SUITE_TOTAL for planned run count)"
        )
    elif config.suite_total is not None:
        lines.append(
            f"Planned suite total: {config.suite_total} "
            "(set EXPERIMENT_SUITE_INDEX for this run's position)"
        )
    if config.suite_note:
        lines.append(f"Suite note: {config.suite_note}")
    lines.append("")
    return lines


def train(
    docs: list[str],
    *,
    tok: Tokeniser,
    config: RunConfig,
) -> tuple[StateDict, float, list[float]]:
    """Train the GPT model on the dataset."""
    config.validate()
    n_layer = config.n_layer
    n_embd = config.n_embd
    n_head = config.n_head
    head_dim = config.head_dim
    block_size = config.block_size
    num_steps = config.num_steps
    learning_rate = config.learning_rate
    beta1 = config.beta1
    beta2 = config.beta2
    eps_adam = config.eps_adam

    state_dict: StateDict = {
        "wte": make_matrix(tok.vocab_size, nin=n_embd),
        "wpe": make_matrix(block_size, nin=n_embd),
        "lm_head": make_matrix(tok.vocab_size, nin=n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = make_matrix(n_embd, nin=n_embd)
        state_dict[f"layer{i}.attn_wk"] = make_matrix(n_embd, nin=n_embd)
        state_dict[f"layer{i}.attn_wv"] = make_matrix(n_embd, nin=n_embd)
        state_dict[f"layer{i}.attn_wo"] = make_matrix(n_embd, nin=n_embd)
        state_dict[f"layer{i}.mlp_fc1"] = make_matrix(4 * n_embd, nin=n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = make_matrix(n_embd, nin=4 * n_embd)

    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"Num Params: {len(params)}")

    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    final_loss = float("nan")
    loss_history: list[float] = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [tok.bos] + [tok.uchars.index(ch) for ch in doc] + [tok.bos]
        seq_len = min(block_size, len(tokens) - 1)

        kv_keys: KVCache = [[] for _ in range(n_layer)]
        kv_values: KVCache = [[] for _ in range(n_layer)]
        losses: Vector = []
        for pos_id in range(seq_len):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(
                token_id,
                pos_id=pos_id,
                keys=kv_keys,
                values=kv_values,
                state=state_dict,
                n_layer=n_layer,
                n_head=n_head,
                head_dim=head_dim,
            )
            probs = softmax(logits)
            losses.append(-probs[target_id].log())

        loss = (1 / seq_len) * sum(losses, Value(0.0))
        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad**2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0.0

        final_loss = loss.data
        loss_history.append(float(loss.data))
        if step >= 100:
            loss_avg_100 = sum(loss_history[-100:]) / 100
            print(
                f"Step {step + 1:4d} / {num_steps:4d} | Loss {loss.data:.4f} | "
                f"Avg-100 {loss_avg_100:.4f}",
                end="\r",
            )
        else:
            print(
                f"Step {step + 1:4d} / {num_steps:4d} | Loss {loss.data:.4f}",
                end="\r",
            )
    return state_dict, final_loss, loss_history


def generate(
    state: StateDict,
    *,
    tok: Tokeniser,
    config: RunConfig,
) -> list[str]:
    """Sample 20 names from a trained model."""
    config.validate()
    n_layer = config.n_layer
    n_head = config.n_head
    head_dim = config.head_dim
    block_size = config.block_size
    temperature = config.temperature

    samples: list[str] = []
    for _sample_idx in range(20):
        kv_keys: KVCache = [[] for _ in range(n_layer)]
        kv_values: KVCache = [[] for _ in range(n_layer)]
        token_id = tok.bos
        sample: list[str] = []
        for pos_id in range(block_size):
            logits = gpt(
                token_id,
                pos_id=pos_id,
                keys=kv_keys,
                values=kv_values,
                state=state,
                n_layer=n_layer,
                n_head=n_head,
                head_dim=head_dim,
            )
            probs = softmax([logit / temperature for logit in logits])
            token_id = random.choices(
                range(tok.vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == tok.bos:
                break
            sample.append(tok.uchars[token_id])
        samples.append("".join(sample))
    return samples


def save_run_report(
    path: Path,
    *,
    config: RunConfig,
    final_loss: float,
    samples: list[str],
    loss_history: list[float],
    char_dist_score: float | None = None,
    quality_metrics: dict[str, float] | None = None,
    semantic_quality: dict[str, object] | None = None,
) -> None:
    """Write hyperparameters, final training loss, and generated lines to a file."""
    lines = build_run_report_lines(
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_head=config.n_head,
        block_size=config.block_size,
        num_steps=config.num_steps,
        temperature=config.temperature,
        seed=config.seed,
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        eps_adam=config.eps_adam,
        input_path=config.input_path,
        final_loss=final_loss,
        samples=samples,
        loss_history=loss_history,
        experiment_suite_lines=experiment_suite_lines(config),
        char_dist_score=char_dist_score,
        quality_metrics=quality_metrics,
        semantic_quality=semantic_quality,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    config: RunConfig,
    *,
    output_dir: Path | None = None,
    save_report: bool = True,
    print_samples: bool = True,
    print_quality: bool = True,
) -> ExperimentResult:
    """Load data, train, generate, evaluate, and optionally save a run report."""
    config.validate()
    random.seed(config.seed)

    docs = load_dataset(input_path=config.input_path, names_url=config.names_url)
    tok = build_tokeniser(docs)

    state_dict, final_loss, loss_history = train(docs, tok=tok, config=config)
    samples = generate(state_dict, tok=tok, config=config)

    if print_samples:
        print("\n--- Inference (new, hallucinated names) ---")
        for i, name in enumerate(samples, start=1):
            print(f"Sample {i:2d}: {name}")

    char_dist_score, quality_metrics, semantic_quality = compute_sample_quality_metrics(
        samples, docs
    )
    if print_quality:
        for line in format_sample_quality_console_lines(
            char_dist_score,
            quality_metrics,
            semantic_quality,
            n_samples=len(samples),
        ):
            print(line)

    report_path: Path | None = None
    if save_report and output_dir is not None:
        report_path = format_run_output_path_for_params(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            num_steps=config.num_steps,
            temperature=config.temperature,
            seed=config.seed,
            directory=output_dir,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_run_report(
            report_path,
            config=config,
            final_loss=final_loss,
            samples=samples,
            loss_history=loss_history,
            char_dist_score=char_dist_score,
            quality_metrics=quality_metrics,
            semantic_quality=semantic_quality,
        )
        print(f"\nSaved run report to {report_path.resolve()}")

    return ExperimentResult(
        config=config,
        final_loss=final_loss,
        samples=samples,
        loss_history=loss_history,
        char_dist_score=char_dist_score,
        quality_metrics=quality_metrics,
        semantic_quality=semantic_quality,
        report_path=report_path,
    )
