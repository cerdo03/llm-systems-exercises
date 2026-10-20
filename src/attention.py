"""Readable, single-head scaled dot-product attention."""

import numpy as np


def scaled_dot_product_attention(Q, K, V, *, causal=False):
    """Return [T_q, d_v] attention output using float64 computation.

    Q: [T_q, d_k], K: [T_k, d_k], V: [T_k, d_v].
    Inputs must be finite floating-point arrays with positive dimensions.
    Causal mode requires aligned, equal-length query/key sequences.
    Query i can attend to key j exactly when j <= i.
    """
    arrays = []
    for name, value in (("Q", Q), ("K", K), ("V", V)):
        array = np.asarray(value)
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name} must be a nonempty 2D array")
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{name} must contain floating-point values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(array.astype(np.float64, copy=False))
    Q, K, V = arrays
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must have the same vector width")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V must have the same number of tokens")
    if causal and Q.shape[0] != K.shape[0]:
        raise ValueError("Causal attention requires equal query/key lengths")

    scores = (Q @ K.T) / np.sqrt(Q.shape[1])  # [T_q, T_k]
    if causal:
        future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)

    # Subtract each row's maximum before exponentiation (stable softmax).
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return weights @ V  # [T_q, d_v]
