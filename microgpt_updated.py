"""
Train and run inference for a GPT in pure, dependency-free Python.

The most atomic version of the algorithm. This file is it.
Everything else is just efficiency.

@karpathy
https://karpathy.github.io/2026/02/12/microgpt/
"""

from __future__ import annotations

import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated, NamedTuple

# Hyperparameters
# ====================

# Depth of the transformer (number of layers). 1 is enough for
# character-level name generation; deeper nets help for harder tasks but
# cost linearly more here because our scalar autograd is O(nodes) per
# backward pass and each layer adds a fixed number of nodes.
N_LAYER = 1

# Width of the network (embedding dimension). Each token and position is
# represented as a vector of this length. 16 is tiny but trains fast in
# pure Python and is enough to learn spelling patterns in short names.
N_EMBD = 16

# Maximum context length of the attention window. The longest name in the
# dataset is 15 characters, so 16 covers every example with room for BOS.
BLOCK_SIZE = 16

# Number of attention heads. Multi-head attention lets the model attend to
# different positional relationships in parallel. 4 heads of dimension 4
# (16 / 4) is a reasonable split for this embedding size.
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

    Kept in one place so :func:`save_run_report` and
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
    t_token = f"{TEMPERATURE:g}".replace("-", "m").replace(".", "p")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    head_dim = N_EMBD // N_HEAD
    stem = (
        f"{prefix}_L{N_LAYER}_E{N_EMBD}_H{N_HEAD}_D{head_dim}_B{BLOCK_SIZE}"
        f"_S{NUM_STEPS}_T{t_token}_seed{SEED}_{run_ts}"
    )
    return Path(directory) / f"{stem}.txt"


def save_run_report(
    path: Path,
    *,
    final_loss: float,
    samples: list[str],
) -> None:
    """Write hyperparameters, final training loss, and generated lines to a file."""
    head_dim = N_EMBD // N_HEAD
    lines = [
        "microGPT run report",
        "===================",
        *format_run_narrative_lines(
            n_layer=N_LAYER,
            n_embd=N_EMBD,
            n_head=N_HEAD,
            head_dim=head_dim,
            block_size=BLOCK_SIZE,
            num_steps=NUM_STEPS,
            temperature=TEMPERATURE,
            seed=SEED,
            input_path=INPUT_PATH,
            final_loss=final_loss,
            num_samples=len(samples),
        ),
        *_experiment_suite_lines(),
        "--- Config (this run) ---",
        f"N_LAYER={N_LAYER}",
        f"N_EMBD={N_EMBD}",
        f"N_HEAD={N_HEAD}",
        f"HEAD_DIM={head_dim}",
        f"BLOCK_SIZE={BLOCK_SIZE}",
        f"NUM_STEPS={NUM_STEPS}",
        f"TEMPERATURE={TEMPERATURE}",
        f"SEED={SEED}",
        f"LEARNING_RATE={LEARNING_RATE}",
        f"BETA1={BETA1}",
        f"BETA2={BETA2}",
        f"EPS_ADAM={EPS_ADAM}",
        f"INPUT_PATH={INPUT_PATH}",
        "",
        f"Final loss (last training step): {final_loss:.6f}",
        "",
        "--- Inference samples ---",
    ]
    for i, name in enumerate(samples, start=1):
        lines.append(f"Sample {i:2d}: {name}")
    lines.append("")
    lines.extend(run_parameter_glossary_lines())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Let there be Autograd: recursively apply the chain rule through a computation graph.
# ------------
# Training requires gradients: for each parameter, we need to know "if I nudge
# this number up a little, does the loss go up or down, and by how much?". In
# production, PyTorch handles this. Here we do it from scratch.
class Value:
    """Scalar node in an autograd computation graph.

    Each Value wraps a single number and tracks how it was computed. Think of
    each operation as a lego block: it takes some inputs, produces an output
    (the forward pass), and knows how its output would change w.r.t. each
    input (the local gradient). That is all autograd needs from each block.
    Everything else is just the chain rule, stringing the blocks together.

    This is the same algorithm that PyTorch's loss.backward() runs, just
    on scalars instead of tensors: algorithmically identical, significantly
    smaller and simpler, but of course a lot less efficient.

    Attributes:
        data: Scalar value of this node, computed during the forward pass.
        grad: Derivative of the loss w.r.t. this node, computed in the backward pass.
    """

    # Memory optimisation: __slots__ avoids a per-instance __dict__
    __slots__ = ("_children", "_local_grads", "data", "grad")

    def __init__(
        self,
        data: float,
        *,
        children: tuple[Value, ...] = (),
        local_grads: tuple[float, ...] = (),
    ) -> None:
        # Scalar value, calculated during the forward pass
        self.data = data

        # d(loss)/d(self), accumulated during the backward pass
        self.grad = 0.0

        # Children of this node in the computation graph
        self._children = children

        # d(self)/d(child) for each child (the local Jacobian entries)
        self._local_grads = local_grads

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            children=(self, other),
            local_grads=(1.0, 1.0),
        )

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            children=(self, other),
            local_grads=(other.data, self.data),
        )

    def __pow__(self, other: float) -> Value:
        return Value(
            self.data**other,
            children=(self,),
            local_grads=(other * self.data ** (other - 1),),
        )

    def log(self) -> Value:
        """Natural logarithm."""
        return Value(
            math.log(self.data),
            children=(self,),
            local_grads=(1.0 / self.data,),
        )

    def exp(self) -> Value:
        """Natural exponential."""
        ex = math.exp(self.data)
        return Value(ex, children=(self,), local_grads=(ex,))

    def relu(self) -> Value:
        """Rectified linear unit."""
        return Value(
            max(0, self.data),
            children=(self,),
            local_grads=(float(self.data > 0),),
        )

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: float) -> Value:
        return other * self**-1

    def backward(self) -> None:
        """Backpropagate gradients through the computation graph.

        Walks the graph in reverse topological order (from loss to
        parameters), applying the chain rule at each step. The chain rule
        is just multiplying rates of change along the path: "if a car
        travels twice as fast as a bicycle and the bicycle is four times
        as fast as a walking man, then the car travels 2 x 4 = 8 times as
        fast as the man."
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # dL/dL = 1: the loss's rate of change w.r.t. itself is trivially 1
        self.grad = 1.0
        for node in reversed(topo):
            for child, local_grad in zip(
                node._children, node._local_grads, strict=True
            ):
                # += not = : when a value is used in multiple places the graph
                # branches, and gradients from each branch must be summed
                # (multivariable chain rule)
                child.grad += local_grad * node.grad


# ==================================================================================
# Define the model architecture: a function mapping tokens and parameters to logits
# Follow GPT-2, blessed among the GPTs, with minor differences:
#   layernorm -> rmsnorm, no biases, GeLU -> ReLU
# ==================================================================================

Vector = Annotated[list[Value], "An embedding or hidden state"]

Matrix = Annotated[list[Vector], "A weight matrix (rows of vectors)"]

KVCache = Annotated[list[list[Vector]], "A [layer][timestep] -> key or value vector"]

StateDict = Annotated[dict[str, Matrix], "Model parameters keyed by name"]


class Tokeniser(NamedTuple):
    """Character-level tokeniser mapping chars to integer token ids."""

    uchars: Annotated[list[str], "sorted unique characters (ids 0..n-1)"]
    bos: Annotated[int, "Beginning of Sequence token id"]
    vocab_size: Annotated[int, "Total unique tokens (len(uchars) + 1)"]


def linear(x: Vector, *, w: Matrix) -> Vector:
    """Matrix-vector multiply, the fundamental building block of neural networks.

    Computes one dot product per row of w: a learned linear transformation.
    """
    return [
        sum((wi * xi for wi, xi in zip(wo, x, strict=True)), Value(0.0)) for wo in w
    ]


def softmax(logits: Vector) -> Vector:
    """Convert raw scores (logits) into a probability distribution.

    All values end up in [0, 1] and sum to 1. Subtracting the max first
    does not change the result mathematically but prevents overflow in exp.
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


def rmsnorm(x: Vector) -> Vector:
    """Root mean square layer normalisation.

    Rescales a vector so its values have unit root-mean-square. This keeps
    activations from growing or shrinking as they flow through layers,
    which stabilises training. Simpler variant of GPT-2's LayerNorm.
    """
    mean_sq = sum((xi * xi for xi in x), Value(0.0)) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def make_matrix(nout: int, *, nin: int, std: float = 0.08) -> Matrix:
    """Create a (nout x nin) matrix of randomly initialised Value nodes."""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def gpt(
    token_id: int,
    *,
    pos_id: int,
    keys: KVCache,
    values: KVCache,
    state: StateDict,
) -> Vector:
    """Forward pass for a single token through the GPT model.

    Processes one token at a time (no batch dimension, no parallel time
    steps), building up the KV cache explicitly. Unlike the typical
    inference setting where the KV cache holds detached tensors, here the
    cached keys and values are live Value nodes in the computation graph,
    so we actually backpropagate through them.
    """
    # The neural network cannot process a raw token id like 5 directly. It
    # can only work with vectors. The token and position each look up a row
    # from their embedding tables, then add together to encode both "what
    # the token is" and "where it sits in the sequence".
    tok_emb = state["wte"][token_id]

    # Modern LLMs usually skip learned position embeddings in favour of
    # relative schemes like RoPE, but absolute position embeddings are
    # simpler and sufficient for short sequences.
    pos_emb = state["wpe"][pos_id]

    # Add the two vectors together, giving the model a representation that
    # encodes both what the token is and where it is in the sequence.
    x = [t + p for t, p in zip(tok_emb, pos_emb, strict=True)]

    # Not redundant: residual connection needs normalised input for backward
    x = rmsnorm(x)

    for li in range(N_LAYER):
        # 1) Multi-head Attention block
        # The ONLY place where a token at position t gets to "look at"
        # tokens in the past 0..t-1. Attention is a communication mechanism.
        # Q = "what am I looking for?"
        # K = "what do I contain?"
        # V = "what do I offer if selected?"
        # e.g. in "emma", when at the second "m" trying to predict what
        # comes next, the query might encode "what vowels appeared
        # recently?". The earlier "e" has a key that matches well, so its
        # value (information about being a vowel) flows in.
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, w=state[f"layer{li}.attn_wq"])
        k = linear(x, w=state[f"layer{li}.attn_wk"])
        v = linear(x, w=state[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn: Vector = []

        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs : hs + HEAD_DIM]
            k_h = [ki[hs : hs + HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs : hs + HEAD_DIM] for vi in values[li]]

            # Dot products between query and all cached keys, scaled by
            # sqrt(d_head) to keep variance stable
            attn_logits = [
                sum(
                    (q_h[j] * k_h[t][j] for j in range(HEAD_DIM)),
                    Value(0.0),
                )
                / HEAD_DIM**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)

            # Weighted sum of cached values
            head_out = [
                sum(
                    (attn_weights[t] * v_h[t][j] for t in range(len(v_h))),
                    Value(0.0),
                )
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)

        # Concatenated head outputs are projected through attn_wo
        x = linear(x_attn, w=state[f"layer{li}.attn_wo"])

        # Residual: lets gradients flow directly through, making deeper
        # models trainable
        x = [a + b for a, b in zip(x, x_residual, strict=True)]

        # 2) MLP block (two-layer feed-forward network)
        # Project up to 4x the embedding dimension, apply ReLU, project
        # back down. This is where the model does most of its "thinking"
        # per position. Unlike attention, fully local to time t. The
        # Transformer intersperses communication (Attention) with
        # computation (MLP).
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, w=state[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, w=state[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual, strict=True)]

    # Project to vocab_size logits: one score per possible next token
    # (in our case, just 27 numbers). Higher logit = more likely next.
    return linear(x, w=state["lm_head"])


def load_dataset() -> list[str]:
    """Let there be a Dataset: list[str] of documents (e.g. a list of names).

    In production each document would be an internet web page. Here we use
    32,000 names, one per line. The goal is to learn the patterns and then
    generate similar new documents. From ChatGPT's perspective, your
    conversation is just a funny looking "document" and its response is
    just a statistical completion.
    """
    if not os.path.exists(INPUT_PATH):
        import urllib.request

        urllib.request.urlretrieve(NAMES_URL, INPUT_PATH)

    with open(INPUT_PATH) as f:
        docs = [line.strip() for line in f if line.strip()]

    random.shuffle(docs)
    print(f"Num Docs: {len(docs)}")
    return docs


def build_tokeniser(docs: list[str]) -> Tokeniser:
    """Let there be a Tokeniser: strings to sequences of integer tokens and back.

    Neural networks work with numbers, not characters, so we assign one
    integer to each unique character. The integer values have no meaning;
    each token is just a separate discrete symbol. Production tokenisers
    like tiktoken (GPT-4) operate on chunks of characters for efficiency,
    but character-level is the simplest possible scheme.

    Returns:
        uchars: Sorted unique characters (token ids 0..n-1).
        bos: Beginning of Sequence token id.
        vocab_size: Total number of unique tokens.
    """
    # unique chars become token ids 0..n-1
    uchars = sorted(set("".join(docs)))

    # BOS acts as a delimiter: "a new document starts/ends here". During
    # training each name is wrapped: [BOS, e, m, m, a, BOS]. The model
    # learns that BOS initiates a new name and another BOS ends it.
    bos = len(uchars)

    # 26 lowercase a-z + 1 BOS = 27
    vocab_size = len(uchars) + 1
    print(f"Vocab Size: {vocab_size}")
    return Tokeniser(uchars, bos, vocab_size)


def train(
    docs: list[str],
    *,
    tok: Tokeniser,
) -> tuple[StateDict, float]:
    """Train the GPT model on the dataset.

    Each step:
        (1) pick a document
        (2) run the model forward over its tokens
        (3) compute a loss
        (4) backpropagate to get gradients
        (5) update the parameters.

    The model's job is to predict each next token given the tokens before it.

    Returns:
        Trained parameters and the loss value from the final optimisation step.
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
        print(
            f"Step {step + 1:4d} / {NUM_STEPS:4d} | Loss {loss.data:.4f}",
            end="\r",
        )
    return state_dict, final_loss


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
    docs = load_dataset()

    # Map characters to integer token ids (a=0 .. z=25, BOS=26).
    tok = build_tokeniser(docs)

    # Iterate over documents, nudging parameters to reduce prediction
    # error. After this, the statistical patterns of names are distilled
    # into the model's weights.
    state_dict, final_loss = train(docs, tok=tok)

    # Freeze parameters and sample new names by feeding each generated
    # token back as the next input.
    samples = generate(state_dict, tok=tok)
    print("\n--- Inference (new, hallucinated names) ---")
    for i, name in enumerate(samples, start=1):
        print(f"Sample {i:2d}: {name}")

    out_path = format_run_output_path()
    save_run_report(out_path, final_loss=final_loss, samples=samples)
    print(f"\nSaved run report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
