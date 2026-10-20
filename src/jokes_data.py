"""Clean Reddit r/Jokes into one joke per record.

Rules (agreed 2026-10-18):
1. keep score >= MIN_SCORE
2. drop empty, [removed], [deleted] bodies
3. join as "title\\nbody"
4. normalize to ASCII; drop jokes that still contain other characters
5. drop jokes longer than MAX_CHARS (setup must fit in the context window)
6. drop exact duplicates (after normalization)
No NSFW filter yet.

Output is JSON Lines (one JSON string per line) so joke boundaries are kept
without deciding the training format yet (milestone 3).

Usage: python -m src.jokes_data data/reddit_jokes.json data/jokes_clean.jsonl [min_score]
"""

import json
import sys
import unicodedata
from collections import Counter

MIN_SCORE = 50
MAX_CHARS = 512
DEAD_BODIES = {"", "[removed]", "[deleted]"}

# Common typographic characters that NFKD does not turn into ASCII.
REPLACEMENTS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ", "\t": " ",
})


def normalize(text):
    """Return ASCII text, or None if a non-ASCII character remains."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(REPLACEMENTS)
    # NFKD splits accented letters: "é" -> "e" + combining accent; drop the accent.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    if any(ord(c) > 127 or (c < " " and c != "\n") for c in text):
        return None
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def clean(raw_jokes, min_score=MIN_SCORE, max_chars=MAX_CHARS):
    """Return (kept joke strings in input order, Counter of drop reasons)."""
    kept, seen, dropped = [], set(), Counter()
    for joke in raw_jokes:
        if joke["score"] < min_score:
            dropped["low_score"] += 1
            continue
        if joke["body"].strip() in DEAD_BODIES:
            dropped["dead_body"] += 1
            continue
        text = normalize(joke["title"].strip() + "\n" + joke["body"].strip())
        if text is None:
            dropped["non_ascii"] += 1
        elif len(text) > max_chars:
            dropped["too_long"] += 1
        elif text in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(text)
            kept.append(text)
    return kept, dropped


def main(src, dst, min_score=MIN_SCORE):
    with open(src) as f:
        raw = json.load(f)
    kept, dropped = clean(raw, min_score=int(min_score))
    with open(dst, "w") as f:
        f.writelines(json.dumps(text) + "\n" for text in kept)
    print(f"input {len(raw)}  kept {len(kept)}  chars {sum(map(len, kept))}")
    for reason, n in dropped.most_common():
        print(f"  dropped {reason}: {n}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
