"""Copy-metric and sampling-knob tests with hand-computed answers."""

import torch

from src import joke_eval as je
from src import joke_gpt as jg
from src.joke_dataset import SEP

TRAIN = [[1, 2, 3, 4, 5, 6, 7], [9, 9, 2, 3, 4, 5, 8]]
INDEX = je.build_index(TRAIN)


def test_longest_run_finds_full_overlap():
    # gen shares 2,3,4,5,6,7 (6 tokens) with joke 0.
    assert je.longest_shared_run([0, 2, 3, 4, 5, 6, 7, 0], TRAIN, INDEX) == 6


def test_longest_run_across_two_jokes_takes_best_single_joke():
    # 2,3,4,5 appears in both; continuing with 8 matches only joke 1 -> 5 tokens.
    assert je.longest_shared_run([2, 3, 4, 5, 8], TRAIN, INDEX) == 5


def test_no_run_of_four_returns_zero():
    assert je.longest_shared_run([1, 2, 3, 0, 5, 6], TRAIN, INDEX) == 0


def test_top_k_1_is_greedy_and_deterministic():
    torch.manual_seed(0)
    model = jg.JokeGPT(jg.Config(block_size=16, n_embedd=16, num_heads=2, n_layer=1, dropout=0.0)).eval()
    a = model.generate(max_new_tokens=8, top_k=1, generator=torch.Generator().manual_seed(1))
    b = model.generate(max_new_tokens=8, top_k=1, generator=torch.Generator().manual_seed(2))
    assert a == b and SEP not in a
