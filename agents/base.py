"""The Agent contract. The M1 model will conform to this same interface."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from wordle.env import Obs


class Agent(Protocol):
    name: str

    def act(self, obs: Obs) -> np.ndarray:
        """Return [B] allowed-word ids — one guess per active game."""
        ...
