"""One test per cleaning rule agreed with the learner."""

from src import jokes_data


def joke(title="Why?", body="Because.", score=100):
    return {"title": title, "body": body, "score": score, "id": "x"}


def test_score_threshold_is_inclusive():
    kept, dropped = jokes_data.clean([joke(score=49), joke(body="B", score=50)])
    assert kept == ["Why?\nB"]
    assert dropped["low_score"] == 1


def test_dead_bodies_dropped():
    raw = [joke(body=""), joke(body="  "), joke(body="[removed]"), joke(body="[deleted]")]
    kept, dropped = jokes_data.clean(raw)
    assert kept == []
    assert dropped["dead_body"] == 4


def test_title_and_body_joined_with_newline():
    kept, _ = jokes_data.clean([joke(title=" Setup ", body="Punchline \r\nline2 ")])
    assert kept == ["Setup\nPunchline\nline2"]


def test_typography_and_accents_become_ascii():
    kept, _ = jokes_data.clean([joke(title="“Café”", body="it’s…")])
    assert kept == ['"Cafe"\nit\'s...']


def test_unconvertible_characters_dropped():
    kept, dropped = jokes_data.clean([joke(body="\U0001F602"), joke(body="你好")])
    assert kept == []
    assert dropped["non_ascii"] == 2


def test_length_cap_counts_joined_text():
    # "Why?\n" is 5 chars, so body of 7 -> 12 total.
    kept, dropped = jokes_data.clean(
        [joke(body="a" * 7), joke(body="b" * 8)], max_chars=12
    )
    assert kept == ["Why?\n" + "a" * 7]
    assert dropped["too_long"] == 1


def test_duplicates_dropped_after_normalization():
    kept, dropped = jokes_data.clean([joke(body="it’s"), joke(body="it's")])
    assert kept == ["Why?\nit's"]
    assert dropped["duplicate"] == 1
