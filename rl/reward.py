"""Reward = terminal (win, speed-scaled) + potential-based candidate-reduction shaping.

Computed in the rollout from env state, not in the env (env keeps a zero placeholder).
All functions are vectorized over the batch and operate on numpy arrays.
"""
from __future__ import annotations

import numpy as np

from wordle.feedback_table import encode_pattern


class CandidateTracker:
    """Tracks, per game, which answers are still consistent with all feedback so far.

    `pat_sub` = PAT[:, answer_subset] of shape [n_allowed, A_sub] (uint8 patterns).
    """

    def __init__(self, pat_sub: np.ndarray):
        self.pat_sub = np.asarray(pat_sub)
        self.A = self.pat_sub.shape[1]

    def reset(self, B: int) -> "CandidateTracker":
        self.cand = np.ones((B, self.A), dtype=bool)
        return self

    def update(self, actions: np.ndarray, patterns: np.ndarray, active=None) -> None:
        """actions [B] guess word-ids; patterns [B] observed base-3 feedback ids.

        Only games with `active[b]` True are updated (others are finished)."""
        consistent = self.pat_sub[actions] == patterns[:, None]
        if active is None:
            self.cand &= consistent
        else:
            self.cand[active] &= consistent[active]

    def counts(self) -> np.ndarray:
        return np.maximum(self.cand.sum(axis=1), 1)   # the true secret is always consistent


def observed_pattern(feedback_row: np.ndarray) -> np.ndarray:
    """[B,5] feedback codes for the just-played guess -> [B] base-3 pattern id."""
    return encode_pattern(feedback_row)


def step_reward(won, k, prev_counts, new_counts, reward_cfg, shaping_coef, gamma):
    """Per model-step reward [B]: terminal (win, speed-scaled) + potential-based shaping.

    won [B] bool; k [B] guesses-used (only used where won); prev/new_counts [B] candidate
    counts before/after this guess.
    """
    won = np.asarray(won)
    term = np.where(won, reward_cfg.win_base + reward_cfg.win_speed * (6 - np.asarray(k)),
                    reward_cfg.loss)
    phi_prev = -np.log2(prev_counts)
    phi_new = -np.log2(new_counts)
    shaping = shaping_coef * (gamma * phi_new - phi_prev)
    return (term + shaping).astype(np.float32)
