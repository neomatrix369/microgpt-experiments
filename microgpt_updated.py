"""
Train and run inference for a GPT in pure, dependency-free Python.

Entry script: hyperparameters, training loop, generation, and run-report wiring.
The scalar autograd ``Value``, transformer forward, and data helpers live in
``mgpt``; ``output_*.txt`` format and parsing live in ``run_report``.

@karpathy
https://karpathy.github.io/2026/02/12/microgpt/
"""

from __future__ import annotations

import random
from pathlib import Path

from mgpt.data import build_tokeniser, load_dataset
from mgpt.evaluation import (
    compute_sample_quality_metrics,
    format_sample_quality_console_lines,
)
from mgpt.model import KVCache, StateDict, Tokeniser, gpt
from mgpt.ops import Vector, make_matrix, softmax
from mgpt.value import Value
from run_report import build_run_report_lines, format_run_output_path_for_params

# Hyperparameters
# ====================

# Depth of the transformer (number of layers). 1 is enough for
# character-level name generation; deeper nets help for harder tasks but
# cost linearly more here because our scalar autograd is O(nodes) per
# backward pass and each layer adds a fixed number of nodes.
# tunable parameter
N_LAYER = 1

# Width of the network (embedding dimension). Each token and position is
# represented as a vector of this length. 16 is tiny but trains fast in
# pure Python and is enough to learn spelling patterns in short names.
# tunable parameter
N_EMBD = 16

# Maximum context length of the attention window. The longest name in the
# dataset is 15 characters, so 16 covers every example with room for BOS.
# tunable parameter (fixed for this dataset)
BLOCK_SIZE = 16

# Number of attention heads. Multi-head attention lets the model attend to
# different positional relationships in parallel. 4 heads of dimension 4
# (16 / 4) is a reasonable split for this embedding size.
# tunable parameter (fixed by the embedding size and number of heads)
N_HEAD = 4

# Derived dimension of each attention head.
HEAD_DIM = N_EMBD // N_HEAD

# Initial learning rate for Adam. 0.01 is on the high side for larger
# models but works well here because the model is small and we apply
# linear decay, so the effective rate drops to zero by the final step.
LEARNING_RATE = 0.01

# Adam first-moment decay (beta1). Controls how much the optimiser trusts
# the current gradient vs the running average. Standard default is 0.9;
# 0.85 forgets faster, which helps on a tiny noisy dataset where stale
# momentum would overshoot.
BETA1 = 0.85

# Adam second-moment decay (beta2). Controls the running average of
# squared gradients used to scale the step size per-parameter. 0.99 is
# slightly more aggressive than the typical 0.999, giving faster
# adaptation at the cost of noisier variance estimates (fine here).
BETA2 = 0.99

# Adam epsilon. Added to the denominator to prevent division by zero when
# a parameter's gradient history is near-zero. 1e-8 is the standard
# default.
EPS_ADAM = 1e-8

# Total number of training steps. Each step processes one document. 1000
# is enough for convergence on this dataset (32k names, vocab of 27
# characters).
NUM_STEPS = 1000

# Sampling temperature in (0, 1]. Lower values sharpen the distribution
# (more conservative, common names), higher values flatten it (more
# creative, weirder names). 0.5 is a good middle ground.
TEMPERATURE = 0.5

SEED = 42
NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
INPUT_PATH = "input.txt"

# Optional labels for variant sweeps and run reports. Set ``EXPERIMENT_SUITE_INDEX``
# to the 1-based position of this run; set ``EXPERIMENT_SUITE_TOTAL`` when you
# know how many runs the suite will contain (both None omits the suite line).
EXPERIMENT_SUITE_INDEX: int | None = None
EXPERIMENT_SUITE_TOTAL: int | None = None
# Optional one-line context for a sweep (printed under ``--- Experiment suite ---``).
EXPERIMENT_SUITE_NOTE: str | None = None


def _experiment_suite_lines() -> list[str]:
    if (
        EXPERIMENT_SUITE_INDEX is None
        and EXPERIMENT_SUITE_TOTAL is None
        and EXPERIMENT_SUITE_NOTE is None
    ):
        return []
    lines = ["--- Experiment suite ---"]
    if (
        EXPERIMENT_SUITE_INDEX is not None
        and EXPERIMENT_SUITE_TOTAL is not None
    ):
        lines.append(
            f"Experiment: {EXPERIMENT_SUITE_INDEX} / {EXPERIMENT_SUITE_TOTAL}"
        )
    elif EXPERIMENT_SUITE_INDEX is not None:
        lines.append(
            f"Experiment index: {EXPERIMENT_SUITE_INDEX} "
            "(set EXPERIMENT_SUITE_TOTAL for planned run count)"
        )
    elif EXPERIMENT_SUITE_TOTAL is not None:
        lines.append(
            f"Planned suite total: {EXPERIMENT_SUITE_TOTAL} "
            "(set EXPERIMENT_SUITE_INDEX for this run's position)"
        )
    if EXPERIMENT_SUITE_NOTE:
        lines.append(f"Suite note: {EXPERIMENT_SUITE_NOTE}")
    lines.append("")
    return lines


def format_run_output_path(
    *,
    prefix: str = "output",
    directory: str | Path = ".",
) -> Path:
    """Build a filesystem-safe path from the current module hyperparameters.

    Example: ``output_L1_E16_H4_D4_B16_S1000_T0p5_seed42_20260422_153045.txt``
    (temperature dots become ``p`` so the stem stays a single clear token;
    trailing ``YYYYMMDD_HHMMSS`` is local wall-clock time for this run).
    """
    return format_run_output_path_for_params(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        head_dim=N_EMBD // N_HEAD,
        block_size=BLOCK_SIZE,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        seed=SEED,
        prefix=prefix,
        directory=directory,
    )


def save_run_report(
    path: Path,
    *,
    final_loss: float,
    samples: list[str],
    loss_history: list[float],
    char_dist_score: float | None = None,
    quality_metrics: dict[str, float] | None = None,
    semantic_quality: dict[str, object] | None = None,
) -> None:
    """Write hyperparameters, final training loss, and generated lines to a file."""
    head_dim = N_EMBD // N_HEAD
    lines = build_run_report_lines(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        num_steps=NUM_STEPS,
        temperature=TEMPERATURE,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        eps_adam=EPS_ADAM,
        input_path=INPUT_PATH,
        final_loss=final_loss,
        samples=samples,
        loss_history=loss_history,
        experiment_suite_lines=_experiment_suite_lines(),
        char_dist_score=char_dist_score,
        quality_metrics=quality_metrics,
        semantic_quality=semantic_quality,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Model forward, autograd, data, and tokeniser live in ``mgpt``; this file keeps
# hyperparameters, the training loop, generation, and run-report wiring.

def train(
    docs: list[str],
    *,
    tok: Tokeniser,
) -> tuple[StateDict, float, list[float]]:
    """Train the GPT model on the dataset.

    Each step:
        (1) pick a document
        (2) run the model forward over its tokens
        (3) compute a loss
        (4) backpropagate to get gradients
        (5) update the parameters.

    The model's job is to predict each next token given the tokens before it.

    Returns:
        Trained parameters, the loss value from the final optimisation step,
        and the per-step loss values (one float per training step, in order).
    """
    # The parameters ARE the knowledge of the model. Each is initialised to
    # a small random Gaussian and iteratively nudged during training. The
    # state_dict organises them into named matrices (borrowing PyTorch's
    # terminology). Our tiny model has 4,192 parameters. GPT-2 had 1.6
    # billion; modern LLMs have hundreds of billions.
    state_dict: StateDict = {
        "wte": make_matrix(tok.vocab_size, nin=N_EMBD),
        "wpe": make_matrix(BLOCK_SIZE, nin=N_EMBD),
        "lm_head": make_matrix(tok.vocab_size, nin=N_EMBD),
    }
    for i in range(N_LAYER):
        state_dict[f"layer{i}.attn_wq"] = make_matrix(N_EMBD, nin=N_EMBD)
        state_dict[f"layer{i}.attn_wk"] = make_matrix(N_EMBD, nin=N_EMBD)
        state_dict[f"layer{i}.attn_wv"] = make_matrix(N_EMBD, nin=N_EMBD)
        state_dict[f"layer{i}.attn_wo"] = make_matrix(N_EMBD, nin=N_EMBD)
        state_dict[f"layer{i}.mlp_fc1"] = make_matrix(4 * N_EMBD, nin=N_EMBD)
        state_dict[f"layer{i}.mlp_fc2"] = make_matrix(N_EMBD, nin=4 * N_EMBD)

    # Flatten all weight matrices into a single Vector for the optimiser
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"Num Params: {len(params)}")

    # Let there be Adam, the blessed optimiser, and its buffers.
    # We could just do p.data -= learning_rate * p.grad (plain gradient descent), but
    # Adam is smarter. m tracks the mean of recent gradients (momentum,
    # like a rolling ball) and v tracks the mean of recent squared gradients
    # (adapting the learning rate per-parameter).
    m_buf = [0.0] * len(params)  # first moment (momentum)
    v_buf = [0.0] * len(params)  # second moment (per-param learning_rate scaling)

    # Repeat in sequence
    final_loss = float("nan")
    loss_history: list[float] = []
    for step in range(NUM_STEPS):
        # Take a single document, tokenise it, surround with BOS on both sides
        doc = docs[step % len(docs)]
        tokens = [tok.bos] + [tok.uchars.index(ch) for ch in doc] + [tok.bos]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # Forward the token sequence through the model, building the computation graph
        kv_keys: KVCache = [[] for _ in range(N_LAYER)]
        kv_values: KVCache = [[] for _ in range(N_LAYER)]
        losses: Vector = []
        for pos_id in range(seq_len):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(
                token_id,
                pos_id=pos_id,
                keys=kv_keys,
                values=kv_values,
                state=state_dict,
                n_layer=N_LAYER,
                n_head=N_HEAD,
                head_dim=HEAD_DIM,
            )
            probs = softmax(logits)

            # Cross-entropy loss: -log p(target). Measures "surprise". If
            # the model assigns p=1.0 to the correct token, loss is 0. If
            # p is near 0, loss goes to +inf. The model is punished for
            # being confidently wrong.
            losses.append(-probs[target_id].log())

        # Average loss over the document. Starts at ~3.3 (random guessing
        # among 27 tokens: -log(1/27) ~ 3.3), decreases toward ~2.37.
        # May yours be low.
        loss = (1 / seq_len) * sum(losses, Value(0.0))

        # One call runs backpropagation through the entire computation graph,
        # from the loss back through softmax, the model, and into every
        # parameter. After this, each parameter's .grad tells us how to
        # change it to reduce the loss.
        loss.backward()

        # Adam update: nudge parameters based on their gradients
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)  # linear lr decay
        for i, p in enumerate(params):
            m_buf[i] = BETA1 * m_buf[i] + (1 - BETA1) * p.grad
            v_buf[i] = BETA2 * v_buf[i] + (1 - BETA2) * p.grad**2

            # Bias correction: m and v are initialised to zero, so early
            # estimates are biased toward zero. These correct for that.
            m_hat = m_buf[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_buf[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + EPS_ADAM)

            # Each step builds a new computation graph, so stale gradients
            # are meaningless. Because backward() uses += (the multivariable
            # chain rule), skipping this would mix two unrelated graphs.
            # Same reason PyTorch loops start with optimizer.zero_grad().
            p.grad = 0.0

        final_loss = loss.data
        loss_history.append(float(loss.data))
        if step >= 100:
            loss_avg_100 = sum(loss_history[-100:]) / 100
            print(
                f"Step {step + 1:4d} / {NUM_STEPS:4d} | Loss {loss.data:.4f} | "
                f"Avg-100 {loss_avg_100:.4f}",
                end="\r",
            )
        else:
            print(
                f"Step {step + 1:4d} / {NUM_STEPS:4d} | Loss {loss.data:.4f}",
                end="\r",
            )
    return state_dict, final_loss, loss_history


def generate(
    state: StateDict,
    *,
    tok: Tokeniser,
) -> list[str]:
    """Inference: may the model babble back to us.

    Parameters are frozen. We start each sample with BOS ("begin a new
    name"), sample a token from the model's output distribution, feed it
    back as the next input, and repeat until the model produces BOS again
    ("I'm done") or we hit the maximum sequence length.

    Returns:
        One string per sampled name (same order as printed by :func:`main`).
    """
    samples: list[str] = []
    for _sample_idx in range(20):
        kv_keys: KVCache = [[] for _ in range(N_LAYER)]
        kv_values: KVCache = [[] for _ in range(N_LAYER)]
        token_id = tok.bos
        sample: list[str] = []
        for pos_id in range(BLOCK_SIZE):
            logits = gpt(
                token_id,
                pos_id=pos_id,
                keys=kv_keys,
                values=kv_values,
                state=state,
                n_layer=N_LAYER,
                n_head=N_HEAD,
                head_dim=HEAD_DIM,
            )
            # Dividing logits by temperature before softmax controls
            # randomness. T=1.0 uses the learned distribution directly.
            # T<1.0 sharpens it (conservative). T->0 always picks the
            # most likely token (greedy). T>1.0 flattens (more diverse).
            probs = softmax([logit / TEMPERATURE for logit in logits])
            token_id = random.choices(
                range(tok.vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == tok.bos:
                break
            sample.append(tok.uchars[token_id])
        samples.append("".join(sample))
    return samples


def main() -> None:
    # Let there be order among chaos
    random.seed(SEED)

    # 32k names, one per line. Each name is a "document".
    docs = load_dataset(input_path=INPUT_PATH, names_url=NAMES_URL)

    # Map characters to integer token ids (a=0 .. z=25, BOS=26).
    tok = build_tokeniser(docs)

    # Iterate over documents, nudging parameters to reduce prediction
    # error. After this, the statistical patterns of names are distilled
    # into the model's weights.
    state_dict, final_loss, loss_history = train(docs, tok=tok)

    # Freeze parameters and sample new names by feeding each generated
    # token back as the next input.
    samples = generate(state_dict, tok=tok)
    print("\n--- Inference (new, hallucinated names) ---")
    for i, name in enumerate(samples, start=1):
        print(f"Sample {i:2d}: {name}")

    char_dist_score, quality_metrics, semantic_quality = compute_sample_quality_metrics(
        samples, docs
    )
    for line in format_sample_quality_console_lines(
        char_dist_score,
        quality_metrics,
        semantic_quality,
        n_samples=len(samples),
    ):
        print(line)

    out_path = format_run_output_path()
    save_run_report(
        out_path,
        final_loss=final_loss,
        samples=samples,
        loss_history=loss_history,
        char_dist_score=char_dist_score,
        quality_metrics=quality_metrics,
        semantic_quality=semantic_quality,
    )
    print(f"\nSaved run report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
