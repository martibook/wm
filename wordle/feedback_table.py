"""Precomputed feedback-pattern table for fast candidate tracking.

PAT[g, a] = the feedback of guess g against answer a, encoded as one base-3 int in
0..242 (uint8). Used to compute |consistent answers| cheaply during training (reward
shaping + diagnostics). ~30 MB for the full 12972 x 2315 table; cached to disk + mmap'd.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .feedback import feedback_batch
from .words import REPO_ROOT, WordList

WEIGHTS = np.array([1, 3, 9, 27, 81], dtype=np.int32)   # base-3 place values
CACHE_DIR = REPO_ROOT / "data" / "cache"


def encode_pattern(codes: np.ndarray) -> np.ndarray:
    """[...,5] feedback codes (0/1/2) -> [...] base-3 pattern id in 0..242 (uint8)."""
    return (codes.astype(np.int32) * WEIGHTS).sum(-1).astype(np.uint8)


def build_pattern_table(wl: WordList) -> np.ndarray:
    """[n_allowed, n_answers] uint8 table of feedback patterns."""
    G, A = wl.n_allowed, wl.n_answers
    table = np.empty((G, A), dtype=np.uint8)
    guesses = wl.allowed_ids                                  # [G,5]
    for a in range(A):
        secrets = np.broadcast_to(wl.answers_ids[a], (G, 5))
        table[:, a] = encode_pattern(feedback_batch(guesses, secrets))
    return table


def load_pattern_table(wl: WordList, cache: bool = True) -> np.ndarray:
    path = CACHE_DIR / f"pattern_{wl.n_allowed}x{wl.n_answers}.npy"
    if cache and path.exists():
        return np.load(path, mmap_mode="r")
    table = build_pattern_table(wl)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, table)
        return np.load(path, mmap_mode="r")
    return table
