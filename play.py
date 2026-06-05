"""`play` — run N Wordle games and watch them, then print a summary report.

Usage (from wm/):
    python play.py --n 100 --agent random --seed 0 [--quiet] [--slow]
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rich.console import Console

from agents.model_agent import ModelAgent
from agents.random_agent import RandomAgent
from config import REPO_ROOT
from eval import game_from_obs, play_games, summarize
from render import board_text, dashboard_panel, summary_renderable
from wordle.env import WordleEnv
from wordle.words import load_wordlist

AGENT_CHOICES = ("random", "model")

SLOW_DELAY_S = 1.0  # per-step pause in --slow mode


def make_agent(args, wl):
    if args.agent == "random":
        return RandomAgent(wl, seed=args.seed)
    if args.agent == "model":
        if args.ckpt:
            return ModelAgent.from_checkpoint(args.ckpt, wl)
        return ModelAgent.untrained(wl, seed=args.seed)
    raise SystemExit(f"unknown agent {args.agent!r}")


def parse_args():
    ap = argparse.ArgumentParser(prog="play", description="Watch an agent play Wordle.")
    ap.add_argument("--n", type=int, default=100, help="number of games (default 100)")
    ap.add_argument("--agent", default="random", choices=list(AGENT_CHOICES))
    ap.add_argument("--ckpt", default=None, help="model checkpoint (for --agent model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opener", default="salet")
    ap.add_argument("--stage-answers", type=int, default=None,
                    help="restrict to a curriculum-stage answer subset (match training)")
    ap.add_argument("--stage-pool", type=int, default=None, help="stage guess-pool size (-1 = full)")
    ap.add_argument("--stage-seed", type=int, default=0, help="stage seed (match training)")
    ap.add_argument("--quiet", action="store_true", help="summary only, no live showcase")
    ap.add_argument("--slow", action="store_true", help="animate tile-by-tile")
    ap.add_argument("--plays-dir", default=None, help="where to save play runs (default wm/plays)")
    ap.add_argument("--no-save", action="store_true", help="don't persist this run")
    return ap.parse_args()


def _run_name(meta) -> str:
    """Folder name = agent (+ checkpoint id), no timestamp. Same checkpoint => same folder."""
    if meta["agent"] != "model":
        return meta["agent"]                      # e.g. "random"
    ckpt = meta.get("ckpt")
    if not ckpt:
        return "model_untrained"
    p = Path(ckpt)
    cid = f"{p.parent.parent.name}_{p.stem}" if p.parent.name == "checkpoints" else p.stem
    return f"model_{cid}"


def save_run(plays_dir, meta, result, games) -> Path:
    """Persist a play run (summary.json + games.jsonl + report.txt).

    Named by agent + checkpoint (no timestamp), so re-running the same checkpoint overwrites.
    """
    d = Path(plays_dir) / _run_name(meta)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        **meta,
        "win_rate": result.win_rate,
        "avg_guesses": result.avg_guesses,
        "distribution": result.distribution,
        "failed_words": result.failed_words,
    }, indent=2, default=str))
    with open(d / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps(asdict(g)) + "\n")
    rec = Console(record=True, width=80, force_terminal=True)
    rec.print(summary_renderable(result, meta["agent"], meta["seed"]))
    (d / "report.txt").write_text(rec.export_text())
    return d


def main():
    args = parse_args()
    wl = load_wordlist()
    env = WordleEnv(wl, opener=args.opener)
    agent = make_agent(args, wl)
    console = Console()

    rng = np.random.default_rng(args.seed)
    if args.stage_answers:
        from config import device as _device
        from rl.curriculum import Stage
        from wordle.feedback_table import load_pattern_table
        pool = None if (args.stage_pool is not None and args.stage_pool < 0) else args.stage_pool
        stage = Stage(wl, load_pattern_table(wl), args.stage_answers, pool, _device(), seed=args.stage_seed)
        if hasattr(agent, "pool_mask"):
            agent.pool_mask = stage.pool_mask          # match the trained guess pool
        cand_idx = stage.answer_subset_idx
    else:
        cand_idx = np.arange(wl.n_answers)
    n = min(args.n, len(cand_idx))
    sampled = rng.choice(cand_idx, size=n, replace=False)
    secret_ids = wl.answer_ids[sampled]

    showcase = (not args.quiet) and console.is_terminal
    if showcase:
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
    else:
        games = play_games(agent, env, secret_ids)

    result = summarize(games, env.max_guesses)
    console.print(summary_renderable(result, agent.name, args.seed))

    if not args.no_save:
        plays_dir = args.plays_dir or (REPO_ROOT / "plays")
        meta = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "agent": agent.name, "ckpt": args.ckpt, "n": len(secret_ids), "seed": args.seed,
            "opener": args.opener, "stage_answers": args.stage_answers,
            "stage_pool": args.stage_pool, "stage_seed": args.stage_seed,
        }
        saved = save_run(plays_dir, meta, result, games)
        console.print(f"[dim]saved play run -> {saved}[/]")


if __name__ == "__main__":
    main()
