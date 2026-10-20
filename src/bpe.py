"""Byte-pair encoding tokenizer, written from scratch.

Base vocabulary: the 256 byte values. Each merge adds one token id
(256, 257, ...) that stands for two existing tokens.

Text is first split into chunks (a word with its leading space, digits,
punctuation, whitespace). Merges never cross chunk boundaries, so " dog"
is one token whether followed by "." or "!".
"""

import json
import re
from collections import Counter

CHUNK_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+(?!\S)|\s+")


def chunks(text):
    return CHUNK_RE.findall(text)


def merge(ids, pair, new_id):
    """Replace each non-overlapping occurrence of pair, scanning left to right."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, num_merges):
    """Return merges {(a, b): new_id} in learned order (dicts keep insertion order)."""
    # Identical chunks are trained once, weighted by how often they occur.
    words = {tuple(c.encode()): n for c, n in Counter(chunks(text)).items()}
    merges = {}
    for i in range(num_merges):
        pairs = Counter()
        for ids, n in words.items():
            for pair in zip(ids, ids[1:]):
                pairs[pair] += n
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # ties: pair counted first wins
        new_id = 256 + i
        merges[best] = new_id
        words = {tuple(merge(ids, best, new_id)): n for ids, n in words.items()}
    return merges


def encode(text, merges):
    out, cache = [], {}
    for chunk in chunks(text):
        if chunk not in cache:
            ids = list(chunk.encode())
            while len(ids) > 1:
                # Apply the earliest-learned merge present, same order as training.
                pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
                if pair not in merges:
                    break
                ids = merge(ids, pair, merges[pair])
            cache[chunk] = ids
        out.extend(cache[chunk])
    return out


def vocab(merges):
    """Return {token_id: bytes}."""
    table = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        table[new_id] = table[a] + table[b]
    return table


def decode(ids, merges):
    table = vocab(merges)
    # A model can emit byte sequences that are not valid UTF-8; show them as U+FFFD.
    return b"".join(table[i] for i in ids).decode("utf-8", errors="replace")


def save(merges, path):
    with open(path, "w") as f:
        json.dump([[a, b] for a, b in merges], f)


def load(path):
    with open(path) as f:
        return {(a, b): 256 + i for i, (a, b) in enumerate(json.load(f))}
