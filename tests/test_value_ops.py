"""Smoke tests for scalar autograd, ops, and tiny GPT forward pass."""

from __future__ import annotations

import math
import unittest

from mgpt.model import gpt
from mgpt.ops import linear, make_matrix, rmsnorm, softmax
from mgpt.value import Value


class TestValueAutograd(unittest.TestCase):
    def test_given_x_squared_when_backward_then_grad_is_2x(self) -> None:
        x = Value(3.0)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad, 6.0, places=5)

    def test_given_branching_graph_when_backward_then_grads_sum(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = a + b
        loss = c + d
        loss.backward()
        # d(loss)/da = b + 1 = 4; d(loss)/db = a + 1 = 3
        self.assertAlmostEqual(a.grad, 4.0, places=5)
        self.assertAlmostEqual(b.grad, 3.0, places=5)

    def test_given_negative_input_when_relu_then_zero_grad(self) -> None:
        x = Value(-1.0)
        y = x.relu()
        y.backward()
        self.assertEqual(y.data, 0.0)
        self.assertEqual(x.grad, 0.0)

    def test_given_positive_input_when_relu_then_unit_grad(self) -> None:
        x = Value(2.0)
        y = x.relu()
        y.backward()
        self.assertAlmostEqual(x.grad, 1.0, places=5)


class TestOps(unittest.TestCase):
    def test_given_logits_when_softmax_then_sums_to_one(self) -> None:
        logits = [Value(1.0), Value(2.0), Value(0.5)]
        probs = softmax(logits)
        total = sum(p.data for p in probs)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_given_vector_when_rmsnorm_then_scale_is_reasonable(self) -> None:
        x = [Value(3.0), Value(4.0)]
        out = rmsnorm(x)
        rms = math.sqrt(sum(v.data * v.data for v in out) / len(out))
        self.assertAlmostEqual(rms, 1.0, places=4)

    def test_given_matrix_when_linear_then_output_length_matches_rows(self) -> None:
        x = [Value(1.0), Value(2.0)]
        w = make_matrix(3, nin=2, std=0.01)
        out = linear(x, w=w)
        self.assertEqual(len(out), 3)


class TestGptForward(unittest.TestCase):
    def test_given_tiny_config_when_forward_then_logits_match_vocab(self) -> None:
        n_embd = 8
        n_head = 2
        head_dim = 4
        n_layer = 1
        vocab_size = 5
        block_size = 4

        state = {
            "wte": make_matrix(vocab_size, nin=n_embd, std=0.01),
            "wpe": make_matrix(block_size, nin=n_embd, std=0.01),
            "lm_head": make_matrix(vocab_size, nin=n_embd, std=0.01),
        }
        state["layer0.attn_wq"] = make_matrix(n_embd, nin=n_embd, std=0.01)
        state["layer0.attn_wk"] = make_matrix(n_embd, nin=n_embd, std=0.01)
        state["layer0.attn_wv"] = make_matrix(n_embd, nin=n_embd, std=0.01)
        state["layer0.attn_wo"] = make_matrix(n_embd, nin=n_embd, std=0.01)
        state["layer0.mlp_fc1"] = make_matrix(4 * n_embd, nin=n_embd, std=0.01)
        state["layer0.mlp_fc2"] = make_matrix(n_embd, nin=4 * n_embd, std=0.01)

        keys: list[list] = [[] for _ in range(n_layer)]
        values: list[list] = [[] for _ in range(n_layer)]
        logits = gpt(
            0,
            pos_id=0,
            keys=keys,
            values=values,
            state=state,
            n_layer=n_layer,
            n_head=n_head,
            head_dim=head_dim,
        )
        self.assertEqual(len(logits), vocab_size)


if __name__ == "__main__":
    unittest.main()
