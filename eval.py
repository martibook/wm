"""Stats engine: play games and summarize results.

`evaluate()` is the quiet, batched measurement function (reused by training later).
`play.py` drives its own per-game loop for the live showcase but reuses `summarize()`
and `game_from_obs()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wordle.env import Obs, WordleEnv
from wordle.feedback import codes_to_str
from wordle.words import letter_ids_to_word


@dataclass
class GameResult:
    secret: str
    won: bool
    num_guesses: int            # guesses played, incl. opener (the Wordle score on a win)
    guesses: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    n: int
    win_rate: float
    avg_guesses: float          # among wins only
    distribution: dict          # {1..6: count, "fail": count}
    failed_words: list[str]


def game_from_obs(obs: Obs, i: int, secret_word: str) -> GameResult:
    n = int(obs.turn[i])
    return GameResult(
        secret=secret_word,
        won=bool(obs.won[i]),
        num_guesses=n,
        guesses=[letter_ids_to_word(obs.guesses[i, r]) for r in range(n)],
        feedbacks=[codes_to_str(obs.feedbacks[i, r]) for r in range(n)],
    )


def summarize(games: list[GameResult], max_guesses: int = 6) -> EvalResult:
    n = len(games)
    dist = {k: 0 for k in range(1, max_guesses + 1)}
    dist["fail"] = 0
    win_guesses = []
    failed = []
    for g in games:
        if g.won:
            dist[g.num_guesses] += 1
            win_guesses.append(g.num_guesses)
        else:
            dist["fail"] += 1
            failed.append(g.secret)
    return EvalResult(
        n=n,
        win_rate=(len(win_guesses) / n) if n else 0.0,
        avg_guesses=(sum(win_guesses) / len(win_guesses)) if win_guesses else 0.0,
        distribution=dist,
        failed_words=failed,
    )


def play_games(agent, env: WordleEnv, secret_ids: np.ndarray) -> list[GameResult]:
    secret_ids = np.asarray(secret_ids, dtype=np.int64)
    obs = env.reset(secret_ids)
    while not env.all_done:
        obs, _, _, _ = env.step(agent.act(obs))
    return [game_from_obs(obs, i, env.wl.word_of(int(secret_ids[i]))) for i in range(len(secret_ids))]


def evaluate(agent, env: WordleEnv, secret_ids: np.ndarray) -> EvalResult:
    return summarize(play_games(agent, env, secret_ids), env.max_guesses)
