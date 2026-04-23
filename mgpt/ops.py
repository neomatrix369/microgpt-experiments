"""Low-level ops on :class:`Value` vectors and matrices."""

from __future__ import annotations

import random
from typing import Annotated

from .value import Value

Vector = Annotated[list[Value], "An embedding or hidden state"]
Matrix = Annotated[list[Vector], "A weight matrix (rows of vectors)"]


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
