"""Pack jokes into one token stream with per-token document and position ids.

Layout (SEP = <|joke|>, id 1024, outside BPE):
    SEP joke0 SEP joke1 ... SEP jokeN-1 SEP
Each SEP starts a document, so it is position 0 of the joke after it.
The last token of a joke has SEP as its target: that teaches "the joke ends here".

Attention may only look backwards within the same joke (document mask), and
positions restart at 0 for each joke, matching generation (prompt = SEP).
"""

import numpy as np

from src import bpe

SEP = 1024
VOCAB_SIZE = SEP + 1


def build_stream(jokes, merges, block_size):
    """Return (tokens, doc_ids, pos_ids) int64 arrays and the number of jokes dropped.

    A joke is dropped when SEP + its tokens exceed block_size (setup would not fit).
    """
    tokens, doc_ids, pos_ids, dropped = [], [], [], 0
    for joke in jokes:
        ids = [SEP, *bpe.encode(joke, merges)]
        if len(ids) > block_size:
            dropped += 1
            continue
        doc = doc_ids[-1] + 1 if doc_ids else 0
        tokens.extend(ids)
        doc_ids.extend([doc] * len(ids))
        pos_ids.extend(range(len(ids)))
    # Trailing SEP: the final joke's last token needs an "end" target.
    tokens.append(SEP)
    doc_ids.append(doc_ids[-1] + 1 if doc_ids else 0)
    pos_ids.append(0)
    return (np.array(tokens), np.array(doc_ids), np.array(pos_ids)), dropped


def get_batch(stream, block_size, batch_size, rng):
    """Return x, y, pos [B, T] arrays and doc [B, T].

    Windows start only at a SEP, so no window begins mid-joke with its setup cut off.
    A joke cut off at the window's end is fine: its visible tokens still see their full prefix.
    """
    tokens, doc_ids, pos_ids = stream
    starts = np.flatnonzero((tokens == SEP) & (np.arange(len(tokens)) + block_size < len(tokens)))
    picks = rng.choice(starts, size=batch_size)
    window = picks[:, None] + np.arange(block_size)
    return tokens[window], tokens[window + 1], pos_ids[window], doc_ids[window]


def document_mask(doc):
    """[B, T] doc ids -> [B, T, T] bool; True where query i may attend to key j.

    allowed[b, i, j] = (j <= i) and doc[b, i] == doc[b, j]
    """
    T = doc.shape[-1]
    causal = np.tril(np.ones((T, T), dtype=bool))
    same_doc = doc[..., :, None] == doc[..., None, :]
    return causal & same_doc
