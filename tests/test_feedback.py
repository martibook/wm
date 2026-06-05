"""Feedback correctness — the key M0 gate. Duplicate-letter handling is the crux."""
import numpy as np
import pytest

from wordle.feedback import feedback_batch, score
from wordle.words import load_wordlist, word_to_letter_ids


# (guess, secret, expected) — hand-verified, focused on duplicates.
CASES = [
    ("crane", "crane", "GGGGG"),   # all green
    ("jumpy", "crane", "BBBBB"),   # all gray (no shared letters)
    ("salet", "crane", "BYBYB"),   # the opener vs crane (matches docs example)
    ("aaaaa", "aback", "GBGBB"),   # guess has 5 a's, answer has 2 -> only the green a's count
    ("there", "eerie", "BBYYG"),   # multiple e's: green first, then yellows from the remainder
    ("babes", "abbey", "YYGGB"),   # repeated b's; greens consume before yellows
]


@pytest.mark.parametrize("guess,secret,expected", CASES)
def test_score_cases(guess, secret, expected):
    assert score(guess, secret) == expected


def test_green_consumes_before_yellow():
    # secret 'basis' has two s's; guess 'sassy' has three (pos 0, 2, 3).
    # pos2 s is green (consumes one), pos0 s takes the one remaining as yellow,
    # and pos3 s must then be gray — there are no s's left.
    assert score("sassy", "basis") == "YGGBB"


def _reference(guess: str, secret: str) -> str:
    """Independent pure-Python implementation of the two-pass rule."""
    g, s = list(guess), list(secret)
    out = ["B"] * 5
    remaining = {}
    for i in range(5):
        if g[i] == s[i]:
            out[i] = "G"
        else:
            remaining[s[i]] = remaining.get(s[i], 0) + 1
    for i in range(5):
        if out[i] == "G":
            continue
        if remaining.get(g[i], 0) > 0:
            out[i] = "Y"
            remaining[g[i]] -= 1
    return "".join(out)


def test_batch_matches_reference_on_random_sample():
    wl = load_wordlist()
    rng = np.random.default_rng(0)
    n = 5000
    gi = rng.integers(0, wl.n_allowed, size=n)
    si = rng.integers(0, wl.n_answers, size=n)

    guesses = wl.allowed_ids[gi]
    secrets = wl.answers_ids[si]
    codes = feedback_batch(guesses, secrets)

    code_to_char = np.array(list("BYG"))
    got = ["".join(code_to_char[c] for c in row) for row in codes]
    expected = [_reference(wl.allowed[gi[k]], wl.answers[si[k]]) for k in range(n)]
    assert got == expected


def test_score_matches_feedback_batch():
    wl = load_wordlist()
    g, s = "salet", "crane"
    via_batch = feedback_batch(word_to_letter_ids(g)[None], word_to_letter_ids(s)[None])[0]
    via_string = "".join("BYG"[c] for c in via_batch)
    assert via_string == score(g, s)
