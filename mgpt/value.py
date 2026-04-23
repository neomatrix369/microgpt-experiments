"""Scalar autograd node."""

from __future__ import annotations

import math


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
