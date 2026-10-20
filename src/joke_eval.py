"""Sample jokes at several temperatures and measure copying from the training set.

Copying metric: the longest run of consecutive tokens a generated joke shares
with any training joke. Short runs ("walks into a bar") are normal; long runs mean copying.

Usage: python -m src.joke_eval data/run_masked_s10_d03 data/jokes_clean_s10.jsonl
"""

import json
import statistics as st
import sys
from collections import defaultdict

import torch

from src import bpe
from src import joke_gpt as jg

SEED_LEN = 4  # index training runs by their first 4 tokens, then extend


def build_index(train_ids):
    """train_ids: list of token lists. Returns {4-gram: [(joke, start), ...]}."""
    index = defaultdict(list)
    for j, ids in enumerate(train_ids):
        for s in range(len(ids) - SEED_LEN + 1):
            index[tuple(ids[s:s + SEED_LEN])].append((j, s))
    return index


def longest_shared_run(gen, train_ids, index):
    """Length of the longest token run in gen that also appears in some training joke."""
    best = 0
    for g in range(len(gen) - SEED_LEN + 1):
        for j, s in index.get(tuple(gen[g:g + SEED_LEN]), ()):
            t, n = train_ids[j], SEED_LEN
            while g + n < len(gen) and s + n < len(t) and gen[g + n] == t[s + n]:
                n += 1
            best = max(best, n)
    return best  # 0 means no shared run of SEED_LEN or more tokens


def main(run, train_data, n=100, temps=(0.5, 0.8, 1.0, 1.2), top_k=None):
    merges = bpe.load("data/bpe_1024.json")
    with open("data/jokes_clean.jsonl") as f:
        _, test_jokes = jg.split_jokes([json.loads(line) for line in f])
    held_out = set(test_jokes)
    with open(train_data) as f:
        train = [j for j in map(json.loads, f) if j not in held_out]
    train_set = set(train)
    train_ids = [bpe.encode(j, merges) for j in train]
    index = build_index(train_ids)

    ck = torch.load(f"{run}.pt")
    model = jg.JokeGPT(jg.Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()

    results = {}
    for temp in temps:
        g = torch.Generator().manual_seed(0)
        jokes = [model.generate(temperature=temp, top_k=top_k, generator=g) for _ in range(n)]
        runs = [longest_shared_run(ids, train_ids, index) for ids in jokes]
        texts = [bpe.decode(ids, merges) for ids in jokes]
        results[temp] = {
            "median_run": st.median(runs),
            "runs>=20": sum(r >= 20 for r in runs),
            "exact_copies": sum(t in train_set for t in texts),
            "median_len": st.median(len(ids) for ids in jokes),
            "samples": texts[:3],
            "most_copied": max(zip(runs, texts))[1],
        }
        r = results[temp]
        print(f"T={temp}: median shared run {r['median_run']} tokens, runs>=20: {r['runs>=20']}/{n}, "
              f"exact copies {r['exact_copies']}, median length {r['median_len']}")
    with open(f"{run}_eval.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
