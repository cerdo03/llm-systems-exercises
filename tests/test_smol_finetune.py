"""Batching/padding checks; no model download needed."""

import math

import numpy as np
import torch

from src import smol_finetune as sf


def test_collate_pads_right_and_masks_labels():
    ids, att, lab = sf.collate([[0, 5, 6, 0], [0, 7, 0]], pad_id=0)
    assert ids.tolist() == [[0, 5, 6, 0], [0, 7, 0, 0]]
    assert att.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert lab.tolist() == [[0, 5, 6, 0], [0, 7, 0, -100]]


def test_length_batches_cover_every_item_once():
    items = [([0] * n, n) for n in np.random.default_rng(0).integers(2, 50, 333)]
    batches = sf.length_batches(items, 16, np.random.default_rng(1))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(333)) and all(len(b) <= 16 for b in batches)


def test_bits_per_char_of_uniform_model():
    # A model that always predicts uniformly over V tokens costs log2(V) bits per predicted token.
    V = 8

    class Uniform(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return type("Out", (), {"logits": torch.zeros(*input_ids.shape, V)})()

    items = [([0, 1, 2, 0], 6), ([0, 3, 0], 2)]  # 3 + 2 predicted tokens, 8 chars
    bpc = sf.bits_per_char(Uniform(), items, pad_id=0, device="cpu")
    assert math.isclose(bpc, 5 * math.log2(V) / 8, rel_tol=1e-6)  # float32 loss
