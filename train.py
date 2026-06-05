"""M1 training entrypoint: PPO self-play on a curriculum stage.

Usage (from wm/):
    python train.py --answers 50 --pool 200 --batch 1024 --iters 150
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from agents.model_agent import ModelAgent
from config import RunConfig, device as get_device
from eval import evaluate
from model.transformer import WordleTransformer
from rl.curriculum import Stage
from rl.logging import RunLogger, git_commit, utc_now
from rl.ppo import ppo_update
from rl.rollout import collect_rollout
from wordle.env import WordleEnv
from wordle.feedback_table import load_pattern_table
from wordle.words import load_wordlist


def anneal(base, frac, do_anneal=True):
    """Full strength for the first half, then linearly to 0 by the end."""
    if not do_anneal or frac < 0.5:
        return base
    return max(0.0, base * (1 - (frac - 0.5) * 2))


def parse_args():
    ap = argparse.ArgumentParser(prog="train")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--answers", type=int, default=None, help="Stage-A answer subset size")
    ap.add_argument("--pool", type=int, default=None, help="guess pool size (-1 = full 12972)")
    ap.add_argument("--batch", type=int, default=None, help="games per iteration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = RunConfig(seed=args.seed)
    if args.iters:
        cfg.total_iters = args.iters
    if args.answers:
        cfg.curriculum.n_answers = args.answers
    if args.pool is not None:
        cfg.curriculum.guess_pool_size = None if args.pool < 0 else args.pool
    if args.batch:
        cfg.ppo.batch_games = args.batch
    if args.eval_every:
        cfg.eval_every = args.eval_every

    dev = get_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    wl = load_wordlist()
    PAT = load_pattern_table(wl)
    env = WordleEnv(wl, opener=cfg.curriculum.opener, max_guesses=cfg.curriculum.max_guesses)
    stage = Stage(wl, PAT, cfg.curriculum.n_answers, cfg.curriculum.guess_pool_size, dev, seed=cfg.seed)
    model = WordleTransformer(cfg.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ppo.lr)

    logger = RunLogger(cfg.run_dir)
    logger.write_config({
        "run_id": logger.run_id, "git_commit": git_commit(), "device": dev,
        "started": utc_now(), **asdict(cfg),
    })
    pool_sz = cfg.curriculum.guess_pool_size or wl.n_allowed
    print(f"run {logger.run_id} | device={dev} | StageA answers={cfg.curriculum.n_answers} "
          f"pool={pool_sz} B={cfg.ppo.batch_games} iters={cfg.total_iters}")

    env_steps = 0
    for it in range(1, cfg.total_iters + 1):
        frac = it / cfg.total_iters
        scoef = anneal(cfg.reward.shaping_coef, frac, cfg.reward.anneal_shaping)

        model.train()
        t0 = time.time()
        batch, rstats = collect_rollout(model, env, stage, cfg.reward, cfg.ppo.gamma,
                                        cfg.ppo.gae_lambda, scoef, dev, rng, cfg.ppo.batch_games)
        pstats = ppo_update(model, opt, batch, cfg.ppo)
        env_steps += rstats["n_transitions"]

        logger.log_metrics({
            "iter": it, "wall_clock": utc_now(), "env_steps": env_steps, "games": cfg.ppo.batch_games,
            "train": {k: rstats[k] for k in ("win_rate", "avg_guesses", "mean_return", "mean_reward")},
            "ppo": {k: pstats[k] for k in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                            "clip_frac", "explained_variance", "grad_norm")},
            "hparams": {"lr": cfg.ppo.lr, "entropy_coef": cfg.ppo.entropy_coef, "shaping_coef": scoef},
            "iter_seconds": round(time.time() - t0, 3),
        })
        if it == 1 or it % 5 == 0:
            print(f"it {it:4d} | win {rstats['win_rate']*100:5.1f}% | ret {rstats['mean_return']:6.2f} "
                  f"| ent {pstats['entropy']:.2f} | kl {pstats['approx_kl']:.3f} "
                  f"| ev {pstats['explained_variance']:.2f} | {pstats['clip_frac']:.2f} clip | "
                  f"{time.time()-t0:.2f}s")

        if it % cfg.eval_every == 0 or it == cfg.total_iters:
            model.eval()
            agent = ModelAgent(model, dev, pool_mask=stage.pool_mask, greedy=True)
            res = evaluate(agent, env, stage.subset_allowed_ids)
            logger.log_eval({
                "iter": it, "wall_clock": utc_now(), "n_answers": int(stage.A),
                "win_rate": res.win_rate, "avg_guesses": res.avg_guesses,
                "guess_distribution": res.distribution, "failed_words": res.failed_words[:50],
            })
            print(f"   [eval] iter {it}: greedy win {res.win_rate*100:.1f}% on {stage.A} answers "
                  f"(avg {res.avg_guesses:.2f})")

        if it % cfg.ckpt_every == 0 or it == cfg.total_iters:
            torch.save(model.state_dict(), logger.dir / "checkpoints" / f"iter_{it}.pt")

    logger.close()
    print("done ->", logger.dir)


if __name__ == "__main__":
    main()
