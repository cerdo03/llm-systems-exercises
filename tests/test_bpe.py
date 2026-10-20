"""BPE tests; first two use the hand-worked example from the learner."""

import pytest

from src import bpe

a, b, c, d = map(ord, "abcd")


def test_first_merge_of_hand_example():
    # Learner derived: first merge "aa", text -> ZabdZabac.
    merges = bpe.train("aaabdaaabac", 1)
    assert merges == {(a, a): 256}
    assert bpe.encode("aaabdaaabac", merges) == [256, a, b, d, 256, a, b, a, c]


def test_merge_does_not_reuse_overlapping_token():
    assert bpe.merge([a, a, a], (a, a), 256) == [256, a]
    assert bpe.merge([a, a, a, a], (a, a), 256) == [256, 256]


def test_training_stops_when_no_pairs_left():
    # "ab" -> one merge makes the single token "ab"; nothing left to pair.
    assert bpe.train("ab", 10) == {(a, b): 256}


def test_merged_tokens_can_merge_again():
    merges = bpe.train("abcabcabc", 2)
    assert merges == {(a, b): 256, (256, c): 257}
    assert bpe.vocab(merges)[257] == b"abc"
    assert bpe.encode("abcabcabc", merges) == [257, 257, 257]


@pytest.mark.parametrize("text", [
    "", "a", "Why did the chicken?\nTo get to the other side!",
    "  two  spaces\n\n", "café \U0001F602 unseen", "123,456!!",
])
def test_chunks_and_round_trip_preserve_text(text):
    merges = bpe.train("the chicken the other side to get the", 20)
    assert "".join(bpe.chunks(text)) == text
    assert bpe.decode(bpe.encode(text, merges), merges) == text


def test_merges_never_cross_words():
    merges = bpe.train("a b a b a b a b. a b!", 30)
    for token in list(bpe.vocab(merges).values())[256:]:
        assert b" " not in token[1:]


def test_save_load_round_trip(tmp_path):
    merges = bpe.train("abcabcabc hello hello", 5)
    path = tmp_path / "merges.json"
    bpe.save(merges, path)
    assert bpe.load(path) == merges
    assert list(bpe.load(path)) == list(merges)
