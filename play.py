"""`play` — run N Wordle games and watch them, then print a summary report.

Usage (from wm/):
    python play.py --n 100 --agent random --seed 0 [--quiet] [--slow]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from rich.console import Console

from agents.random_agent import RandomAgent
from eval import evaluate, game_from_obs, summarize
from render import board_text, dashboard_panel, summary_renderable
from wordle.env import WordleEnv
from wordle.words import load_wordlist

AGENTS = {"random": RandomAgent}

SLOW_DELAY_S = 1.0  # per-step pause in --slow mode


def make_agent(name: str, wl, seed: int):
    if name not in AGENTS:
        raise SystemExit(f"unknown agent {name!r}; choices: {', '.join(AGENTS)}")
    return AGENTS[name](wl, seed=seed)


def parse_args():
    ap = argparse.ArgumentParser(prog="play", description="Watch an agent play Wordle.")
    ap.add_argument("--n", type=int, default=100, help="number of games (default 100)")
    ap.add_argument("--agent", default="random", choices=list(AGENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opener", default="salet")
    ap.add_argument("--quiet", action="store_true", help="summary only, no live showcase")
    ap.add_argument("--slow", action="store_true", help="animate tile-by-tile")
    return ap.parse_args()


def main():
    args = parse_args()
    wl = load_wordlist()
    env = WordleEnv(wl, opener=args.opener)
    agent = make_agent(args.agent, wl, args.seed)
    console = Console()

    rng = np.random.default_rng(args.seed)
    n = min(args.n, wl.n_answers)
    sampled = rng.choice(wl.n_answers, size=n, replace=False)
    secret_ids = wl.answer_ids[sampled]

    showcase = (not args.quiet) and console.is_terminal
    if not showcase:
        result = evaluate(agent, env, secret_ids)
        console.print(summary_renderable(result, agent.name, args.seed))
        return

    from rich.live import Live

    games = []
    with Live(console=console, refresh_per_second=60, transient=True) as live:
        for k, sid in enumerate(secret_ids, start=1):
            secret_word = wl.word_of(int(sid))
            obs = env.reset(np.array([int(sid)]))

            def show():
                board = board_text(obs.guesses[0], obs.feedbacks[0], int(obs.turn[0]))
                live.update(dashboard_panel(board, k, n, games, bool(obs.done[0]),
                                            secret_word, bool(obs.won[0])))

            show()
            if args.slow:
                time.sleep(SLOW_DELAY_S)
            while not env.all_done:
                obs, _, _, _ = env.step(agent.act(obs))
                show()
                if args.slow:
                    time.sleep(SLOW_DELAY_S)
            games.append(game_from_obs(obs, 0, secret_word))
            if not args.slow:
                time.sleep(0.02)

    console.print(summary_renderable(summarize(games, env.max_guesses), agent.name, args.seed))


if __name__ == "__main__":
    main()
