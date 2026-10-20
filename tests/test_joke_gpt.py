"""Model-level checks of the document mask and per-joke positions."""

import numpy as np
import torch

from src import joke_dataset as jd
from src import joke_gpt
from src.joke_dataset import SEP

CFG = joke_gpt.Config(block_size=16, n_embedd=16, num_heads=2, n_layer=2, dropout=0.0)


def logits_for(model, tokens, doc_mask):
    tokens = np.array([tokens])
    doc = np.cumsum(tokens == SEP, axis=1) - 1
    pos = np.array([[i - np.flatnonzero(tokens[0, :i + 1] == SEP)[-1] for i in range(tokens.shape[1])]])
    x, p, m, _ = joke_gpt.batch_inputs(tokens, tokens, pos, doc, doc_mask, "cpu")
    return model(x, p, m)[0][0]


def make_model():
    torch.manual_seed(0)
    return joke_gpt.JokeGPT(CFG).eval()


def test_joke_b_ignores_joke_a_with_doc_mask():
    # Same idea as the causal-invariance test: change joke A, joke B's outputs must not move.
    model = make_model()
    out1 = logits_for(model, [SEP, 5, 6, 7, SEP, 8, 9], doc_mask=True)
    out2 = logits_for(model, [SEP, 1, 2, 3, SEP, 8, 9], doc_mask=True)
    torch.testing.assert_close(out1[4:], out2[4:])


def test_joke_b_depends_on_joke_a_without_doc_mask():
    model = make_model()
    out1 = logits_for(model, [SEP, 5, 6, 7, SEP, 8, 9], doc_mask=False)
    out2 = logits_for(model, [SEP, 1, 2, 3, SEP, 8, 9], doc_mask=False)
    assert not torch.allclose(out1[4:], out2[4:])


def test_packed_joke_matches_joke_alone():
    # Mask + position reset: joke B mid-window behaves exactly like joke B at the start (as in generation).
    model = make_model()
    packed = logits_for(model, [SEP, 5, 6, 7, SEP, 8, 9], doc_mask=True)
    alone = logits_for(model, [SEP, 8, 9], doc_mask=True)
    torch.testing.assert_close(packed[4:], alone)


def test_generate_returns_ids_without_sep():
    out = make_model().generate(max_new_tokens=10, generator=torch.Generator().manual_seed(0))
    assert len(out) <= 10 and SEP not in out


def test_generate_keeps_prompt_and_respects_block_size():
    model = make_model()
    prompt = [5, 6, 7]
    out = model.generate(max_new_tokens=100, generator=torch.Generator().manual_seed(0), prompt_ids=prompt)
    assert out[:3] == prompt and len(out) <= CFG.block_size - 1


def test_split_is_by_whole_jokes_and_disjoint():
    jokes = [f"joke {i}" for i in range(100)]
    train, test = joke_gpt.split_jokes(jokes)
    assert len(test) == 10 and len(train) == 90
    assert set(train).isdisjoint(test) and set(train) | set(test) == set(jokes)


def test_batch_shapes_match_dataset():
    stream, _ = jd.build_stream(["ab", "c", "ba"], {}, block_size=4)
    x, p, m, y = joke_gpt.batch_inputs(*jd.get_batch(stream, 4, 3, np.random.default_rng(0)), True, "cpu")
    assert x.shape == p.shape == y.shape == (3, 4) and m.shape == (3, 4, 4) and m.dtype == torch.bool


def test_device_mask_matches_numpy_reference():
    stream, _ = jd.build_stream(["ab", "c", "ba", "cab"], {}, block_size=6)
    xb, yb, pb, db = jd.get_batch(stream, 6, 8, np.random.default_rng(1))
    _, _, m, _ = joke_gpt.batch_inputs(xb, yb, pb, db, True, "cpu")
    np.testing.assert_array_equal(m.numpy(), jd.document_mask(db))
