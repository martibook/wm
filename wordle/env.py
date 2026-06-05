"""Batched Wordle environment.

A single id space = the 12,972 allowed words. `reset` auto-plays the fixed opener
(turn 1), so an agent's first action is turn 2. Done games ignore further actions.

The env emits a *semantic* board (guess letter-ids + B/Y/G codes). Turning that into
model tokens is a model-side concern (M1), keeping the env framework-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feedback import feedback_batch
from .words import WORD_LEN, WordList


@dataclass
class Obs:
    guesses: np.ndarray    # [B, max_guesses, 5] letter ids (filled up to `turn`)
    feedbacks: np.ndarray  # [B, max_guesses, 5] codes 0/1/2
    turn: np.ndarray       # [B] number of guesses made so far (incl. opener)
    done: np.ndarray       # [B] bool
    won: np.ndarray        # [B] bool


class WordleEnv:
    def __init__(self, wl: WordList, opener: str = "salet", max_guesses: int = 6):
        if opener not in wl.allowed:
            raise ValueError(f"opener {opener!r} is not in the allowed-guess list")
        self.wl = wl
        self.opener = opener
        self.opener_id = wl.id_of(opener)
        self.max_guesses = max_guesses

    def reset(self, secrets: np.ndarray) -> Obs:
        """secrets: [B] allowed-word ids (the secret answers). Plays the opener."""
        secrets = np.asarray(secrets, dtype=np.int64)
        self.B = secrets.shape[0]
        self._secret_letters = self.wl.allowed_ids[secrets]            # [B,5]
        self._guesses = np.zeros((self.B, self.max_guesses, WORD_LEN), dtype=np.int8)
        self._feedbacks = np.zeros((self.B, self.max_guesses, WORD_LEN), dtype=np.int8)
        self._turn = np.zeros(self.B, dtype=np.int64)
        self._done = np.zeros(self.B, dtype=bool)
        self._won = np.zeros(self.B, dtype=bool)
        self._apply(np.full(self.B, self.opener_id, dtype=np.int64))  # turn 1 = opener
        return self._obs()

    def step(self, actions: np.ndarray):
        self._apply(np.asarray(actions, dtype=np.int64))
        reward = np.zeros(self.B)  # M0: reward unused (added in M1)
        return self._obs(), reward, self._done.copy(), {}

    def _apply(self, actions: np.ndarray) -> None:
        active = ~self._done
        if not active.any():
            return
        rows = np.arange(self.B)
        guess_letters = self.wl.allowed_ids[actions]                  # [B,5]
        fb = feedback_batch(guess_letters, self._secret_letters)      # [B,5]
        slot = self._turn
        self._guesses[rows[active], slot[active]] = guess_letters[active]
        self._feedbacks[rows[active], slot[active]] = fb[active]
        self._turn[active] += 1
        win_now = active & (fb == 2).all(axis=1)
        self._won |= win_now
        self._done |= win_now | (active & (self._turn >= self.max_guesses))

    def _obs(self) -> Obs:
        return Obs(self._guesses, self._feedbacks, self._turn.copy(),
                   self._done.copy(), self._won.copy())

    @property
    def all_done(self) -> bool:
        return bool(self._done.all())
