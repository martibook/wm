"""Wordle feedback: the duplicate-correct B/Y/G algorithm.

Codes:  0 = gray (B, absent)   1 = yellow (Y, present, wrong spot)   2 = green (G, correct spot)

Two-pass rule (the correctness core):
  Pass 1 (greens): mark exact-position matches; those answer letters are 'consumed'.
  Pass 2 (yellows): left-to-right, a non-green guess letter is yellow only if an
                    *unconsumed* copy of it remains in the answer (decrement on use),
                    else gray. This is what makes duplicate letters correct.
"""
from __future__ import annotations

import numpy as np

from .words import WORD_LEN, word_to_letter_ids

_CODE_TO_CHAR = {0: "B", 1: "Y", 2: "G"}


def feedback_batch(guesses: np.ndarray, secrets: np.ndarray) -> np.ndarray:
    """Vectorized feedback for a batch.

    guesses, secrets: int arrays [B, 5] of letter ids (0..25).
    returns: int8 array [B, 5] of codes (0/1/2).
    """
    guesses = np.asarray(guesses)
    secrets = np.asarray(secrets)
    if guesses.ndim != 2 or guesses.shape[1] != WORD_LEN:
        raise ValueError(f"guesses must be [B, {WORD_LEN}], got {guesses.shape}")
    if guesses.shape != secrets.shape:
        raise ValueError(f"shape mismatch: {guesses.shape} vs {secrets.shape}")

    B = guesses.shape[0]
    rows = np.arange(B)
    codes = np.zeros((B, WORD_LEN), dtype=np.int8)

    greens = guesses == secrets
    codes[greens] = 2

    # Count answer letters that are NOT consumed by greens.
    counts = np.zeros((B, 26), dtype=np.int16)
    for i in range(WORD_LEN):
        mask = ~greens[:, i]
        np.add.at(counts, (rows[mask], secrets[mask, i]), 1)

    # Yellows, left-to-right, decrementing the available count on each use.
    for i in range(WORD_LEN):
        gi = guesses[:, i]
        eligible = ~greens[:, i] & (counts[rows, gi] > 0)
        codes[eligible, i] = 1
        np.add.at(counts, (rows[eligible], gi[eligible]), -1)

    return codes


def feedback_one(guess_ids: np.ndarray, secret_ids: np.ndarray) -> np.ndarray:
    """Feedback for a single guess/secret pair (length-5 letter-id arrays) -> codes [5]."""
    return feedback_batch(np.asarray(guess_ids)[None, :], np.asarray(secret_ids)[None, :])[0]


def codes_to_str(codes) -> str:
    """[0,1,0,1,0] -> 'BYBYB'."""
    return "".join(_CODE_TO_CHAR[int(c)] for c in codes)


def score(guess: str, secret: str) -> str:
    """Human-friendly feedback string, e.g. score('salet', 'crane') -> 'BYBYB'."""
    return codes_to_str(feedback_one(word_to_letter_ids(guess), word_to_letter_ids(secret)))
