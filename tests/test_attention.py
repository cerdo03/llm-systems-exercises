"""Tests based on expectations discussed with the learner."""

import numpy as np
import pytest

from src import attention


@pytest.mark.parametrize("causal", [False, True])
def test_single_token_returns_value(causal):
    q = np.array([[3.0, -2.0]])
    k = np.array([[1.0, 4.0]])
    v = np.array([[2.0, 5.0, -1.0]])

    output = attention.scaled_dot_product_attention(q, k, v, causal=causal)

    assert output.shape == (1, 3)
    np.testing.assert_allclose(output, v)


def test_future_value_cannot_change_first_causal_output():
    q = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = np.array([[2.0, 1.0], [1.0, 3.0]])
    v = np.array([[2.0, 5.0], [7.0, 11.0]])
    changed_v = v.copy()
    changed_v[1] = [-1000.0, 2000.0]

    before = attention.scaled_dot_product_attention(q, k, v, causal=True)
    after = attention.scaled_dot_product_attention(q, k, changed_v, causal=True)

    np.testing.assert_allclose(before[0], v[0])
    np.testing.assert_allclose(after[0], v[0])


def test_equal_scores_average_values():
    q = np.array([[0.0, 0.0]])
    k = np.array([[2.0, 1.0], [1.0, 3.0]])
    v = np.array([[2.0, 4.0], [6.0, 8.0]])

    output = attention.scaled_dot_product_attention(q, k, v)

    assert output.shape == (1, 2)
    # Zero scores yield equal softmax weights, not zero weights.
    np.testing.assert_allclose(output, [[4.0, 6.0]])


def test_large_equal_scores_remain_finite():
    q = np.array([[1000.0]])
    k = np.array([[1.0], [1.0]])
    v = np.array([[2.0, 4.0], [6.0, 8.0]])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        output = attention.scaled_dot_product_attention(q, k, v)

    np.testing.assert_allclose(output, [[4.0, 6.0]])


def test_scaling_produces_hand_derived_weighted_output():
    # d_k = 4: raw scores [0, 2*log(3)] scale to [0, log(3)].
    # Exponentials [1, 3] give weights [1/4, 3/4].
    q = np.array([[1.0, 0.0, 0.0, 0.0]])
    k = np.array([[0.0, 0.0, 0.0, 0.0],
                  [2.0 * np.log(3.0), 0.0, 0.0, 0.0]])
    v = np.array([[2.0, 4.0], [6.0, 8.0]])

    output = attention.scaled_dot_product_attention(q, k, v)

    np.testing.assert_allclose(output, [[5.0, 7.0]])


@pytest.mark.parametrize(
    "q,k,v,causal,error,message",
    [
        (np.ones(2), np.ones((2, 2)), np.ones((2, 1)), False,
         ValueError, "Q must be a nonempty 2D array"),
        (np.ones((0, 2)), np.ones((2, 2)), np.ones((2, 1)), False,
         ValueError, "Q must be a nonempty 2D array"),
        (np.ones((1, 2), dtype=int), np.ones((2, 2)), np.ones((2, 1)), False,
         TypeError, "Q must contain floating-point values"),
        (np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)), False,
         ValueError, "Q must contain only finite values"),
        (np.ones((1, 1)), np.array([[np.inf]]), np.ones((1, 1)), False,
         ValueError, "K must contain only finite values"),
        (np.ones((1, 2)), np.ones((2, 3)), np.ones((2, 1)), False,
         ValueError, "Q and K must have the same vector width"),
        (np.ones((1, 2)), np.ones((2, 2)), np.ones((3, 1)), False,
         ValueError, "K and V must have the same number of tokens"),
        (np.ones((1, 2)), np.ones((2, 2)), np.ones((2, 1)), True,
         ValueError, "Causal attention requires equal query/key lengths"),
    ],
)
def test_invalid_inputs_raise_clear_errors(q, k, v, causal, error, message):
    with pytest.raises(error, match=message):
        attention.scaled_dot_product_attention(q, k, v, causal=causal)
