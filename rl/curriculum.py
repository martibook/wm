"""Curriculum stage: an answer subset + (optional) restricted guess pool + sliced table."""
from __future__ import annotations

import numpy as np
import torch


class Stage:
    """Holds the active answer subset, secret sampler, guess-pool mask, and sliced table."""

    def __init__(self, wl, PAT, n_answers, guess_pool_size, device, seed=0):
        rng = np.random.default_rng(seed)
        self.A = n_answers
        # answer subset: indices into wl.answers (== columns of PAT)
        self.answer_subset_idx = np.sort(rng.choice(wl.n_answers, n_answers, replace=False))
        self.subset_allowed_ids = wl.answer_ids[self.answer_subset_idx]      # secret allowed-ids
        self.pat_sub = np.ascontiguousarray(PAT[:, self.answer_subset_idx])  # [n_allowed, A] uint8

        # guess pool: the answer subset (so the secret is always reachable) + extras
        if guess_pool_size is None:
            self.pool_mask = None
        else:
            pool = set(int(i) for i in self.subset_allowed_ids)
            for c in rng.permutation(wl.n_allowed):
                if len(pool) >= guess_pool_size:
                    break
                pool.add(int(c))
            mask = np.zeros(wl.n_allowed, dtype=bool)
            mask[list(pool)] = True
            self.pool_mask = torch.from_numpy(mask).to(device)

    def sample_secrets(self, B, rng):
        """Returns (secret allowed-ids [B], secret subset-column [B])."""
        pos = rng.integers(0, self.A, B)
        return self.subset_allowed_ids[pos], pos
