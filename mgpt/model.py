"""Transformer forward (single-token step) and tokeniser type."""

from __future__ import annotations

from typing import Annotated, NamedTuple

from .ops import Matrix, Vector, linear, rmsnorm, softmax
from .value import Value

KVCache = Annotated[list[list[Vector]], "A [layer][timestep] -> key or value vector"]

StateDict = Annotated[dict[str, Matrix], "Model parameters keyed by name"]


class Tokeniser(NamedTuple):
    """Character-level tokeniser mapping chars to integer token ids."""

    uchars: Annotated[list[str], "sorted unique characters (ids 0..n-1)"]
    bos: Annotated[int, "Beginning of Sequence token id"]
    vocab_size: Annotated[int, "Total unique tokens (len(uchars) + 1)"]


def gpt(
    token_id: int,
    *,
    pos_id: int,
    keys: KVCache,
    values: KVCache,
    state: StateDict,
    n_layer: int,
    n_head: int,
    head_dim: int,
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
    # the token is" and "where it sits in the sequence."
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

    for li in range(n_layer):
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

        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]

            # Dot products between query and all cached keys, scaled by
            # sqrt(d_head) to keep variance stable
            attn_logits = [
                sum(
                    (q_h[j] * k_h[t][j] for j in range(head_dim)),
                    Value(0.0),
                )
                / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)

            # Weighted sum of cached values
            head_out = [
                sum(
                    (attn_weights[t] * v_h[t][j] for t in range(len(v_h))),
                    Value(0.0),
                )
                for j in range(head_dim)
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
