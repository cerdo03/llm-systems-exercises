"""Packing and document-mask tests; expectations written by hand."""

import numpy as np

from src import joke_dataset as jd
from src.joke_dataset import SEP

a, b, c = map(ord, "abc")
NO_MERGES = {}  # plain bytes, so token ids are easy to read


def test_stream_layout_doc_and_pos_ids():
    (tokens, doc, pos), dropped = jd.build_stream(["ab", "c"], NO_MERGES, block_size=8)
    assert tokens.tolist() == [SEP, a, b, SEP, c, SEP]
    assert doc.tolist() == [0, 0, 0, 1, 1, 2]
    assert pos.tolist() == [0, 1, 2, 0, 1, 0]
    assert dropped == 0


def test_joke_longer_than_block_dropped():
    # "abc" needs SEP + 3 = 4 tokens; block_size 3 cannot hold it.
    (tokens, _, _), dropped = jd.build_stream(["ab", "abc", "c"], NO_MERGES, block_size=3)
    assert tokens.tolist() == [SEP, a, b, SEP, c, SEP]
    assert dropped == 1


def test_document_mask_toy_window():
    # Window: SEP a b | SEP c   (two jokes)
    doc = np.array([[0, 0, 0, 1, 1]])
    expected = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=bool)
    np.testing.assert_array_equal(jd.document_mask(doc)[0], expected)


def test_single_doc_mask_is_plain_causal():
    doc = np.zeros((1, 4), dtype=int)
    np.testing.assert_array_equal(jd.document_mask(doc)[0], np.tril(np.ones((4, 4), bool)))


def test_batches_start_at_sep_and_targets_shift_by_one():
    stream, _ = jd.build_stream(["ab", "c", "ba", "cc"], NO_MERGES, block_size=4)
    x, y, pos, doc = jd.get_batch(stream, block_size=4, batch_size=16, rng=np.random.default_rng(0))
    assert x.shape == y.shape == pos.shape == doc.shape == (16, 4)
    assert (x[:, 0] == SEP).all() and (pos[:, 0] == 0).all()
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    # Positions restart at every SEP inside a window.
    assert (pos[x == SEP] == 0).all()
