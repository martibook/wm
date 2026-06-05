"""Word lists and id maps for the Wordle game.

Single id space = the 12,972 *allowed* guess words (action ids 0..12971).
Secrets are drawn from the 2,315 *answers*, a subset of `allowed`; we expose their
indices within `allowed` via `answer_ids`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # wm/
DATA_DIR = REPO_ROOT / "data"

WORD_LEN = 5


def word_to_letter_ids(word: str) -> np.ndarray:
    """'crane' -> array([2, 17, 0, 13, 4], int8)  (a=0 .. z=25)."""
    return np.frombuffer(word.encode("ascii"), dtype=np.uint8).astype(np.int8) - ord("a")


def letter_ids_to_word(ids) -> str:
    """array([2, 17, 0, 13, 4]) -> 'crane'."""
    return "".join(chr(int(c) + ord("a")) for c in ids)


@dataclass
class WordList:
    answers: list[str]                # 2315 possible secrets
    allowed: list[str]                # 12972 valid guesses (answers ⊆ allowed)

    def __post_init__(self) -> None:
        self._id = {w: i for i, w in enumerate(self.allowed)}
        # [N, 5] letter-id arrays
        self.allowed_ids = np.stack([word_to_letter_ids(w) for w in self.allowed]).astype(np.int8)
        self.answers_ids = np.stack([word_to_letter_ids(w) for w in self.answers]).astype(np.int8)
        # index of each answer within `allowed` (so secrets live in the same id space as actions)
        self.answer_ids = np.array([self._id[w] for w in self.answers], dtype=np.int64)

    @property
    def n_allowed(self) -> int:
        return len(self.allowed)

    @property
    def n_answers(self) -> int:
        return len(self.answers)

    def id_of(self, word: str) -> int:
        return self._id[word]

    def word_of(self, idx: int) -> str:
        return self.allowed[idx]

    def letters_of(self, idx: int) -> np.ndarray:
        return self.allowed_ids[idx]


def load_wordlist(data_dir: str | Path | None = None) -> WordList:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    answers = (data_dir / "answers.txt").read_text().split()
    allowed = (data_dir / "allowed_guesses.txt").read_text().split()
    return WordList(answers=answers, allowed=allowed)
