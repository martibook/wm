"""Baseline player: uniform-random valid guesses. Used to exercise the M0 harness."""
from __future__ import annotations

import numpy as np

from wordle.env import Obs
from wordle.words import WordList


class RandomAgent:
    name = "random"

    def __init__(self, wl: WordList, seed: int = 0):
        self.wl = wl
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Obs) -> np.ndarray:
        return self.rng.integers(0, self.wl.n_allowed, size=obs.turn.shape[0])
