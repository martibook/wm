"""M1/M2 training entrypoint: PPO self-play with an optional curriculum ramp + resume.

Single stage (default):
    python train.py --answers 200 --pool 800 --iters 300
Full curriculum ramp (A: small subset -> B: full answers -> C: full vocab):
    python train.py --curriculum full
Resume a run:
    python train.py --resume runs/<id>/checkpoints/latest.pt
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
from config import RunConfig, device as get_device, pool_size
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
    """Full strength for the first half of the whole run, then linearly to 0."""
    if not do_anneal or frac < 0.5:
        return base
    return max(0.0, base * (1 - (frac - 0.5) * 2))


def build_phases(args):
    if args.curriculum == "full":
        return [
            {"name": "A", "answers": 200, "pool": 800, "iters": args.a_iters},
            {"name": "B", "answers": 2315, "pool": 2315, "iters": args.b_iters},
            {"name": "C", "answers": 2315, "pool": -1, "iters": args.c_iters},   # -1 => full vocab
        ]
    return [{"name": "S", "answers": args.answers or 200,
             "pool": args.pool if args.pool is not None else 800,
             "iters": args.iters or 300}]


def phase_at(phases, gi):
    """Return (phase_dict, index) for 1-based global iteration gi."""
    acc = 0
    for i, p in enumerate(phases):
        acc += p["iters"]
        if gi <= acc:
            return p, i
    return phases[-1], len(phases) - 1


def parse_args():
    ap = argparse.ArgumentParser(prog="train")
    ap.add_argument("--curriculum", choices=["single", "full"], default="single")
    ap.add_argument("--iters", type=int, default=None, help="single-stage iters")
    ap.add_argument("--answers", type=int, default=None, help="single-stage answer subset")
    ap.add_argument("--pool", type=int, default=None, help="single-stage pool (-1 = full)")
    ap.add_argument("--a-iters", type=int, default=300)
    ap.add_argument("--b-iters", type=int, default=1500)
    ap.add_argument("--c-iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--entropy", type=float, default=None, help="entropy coef (exploration)")
    ap.add_argument("--shaping", type=float, default=None, help="shaping coef (narrowing hints)")
    ap.add_argument("--no-anneal-shaping", action="store_true", help="keep shaping on all run")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = RunConfig(seed=args.seed)
    if args.batch:
        cfg.ppo.batch_games = args.batch
    if args.eval_every:
        cfg.eval_every = args.eval_every
    if args.ckpt_every:
        cfg.ckpt_every = args.ckpt_every
    if args.entropy is not None:
        cfg.ppo.entropy_coef = args.entropy
    if args.shaping is not None:
        cfg.reward.shaping_coef = args.shaping
    if args.no_anneal_shaping:
        cfg.reward.anneal_shaping = False

    dev = get_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    wl = load_wordlist()
    PAT = load_pattern_table(wl)
    env = WordleEnv(wl, opener=cfg.curriculum.opener, max_guesses=cfg.curriculum.max_guesses)
    model = WordleTransformer(cfg.model, wl.allowed_ids).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ppo.lr)

    phases = build_phases(args)
    total = sum(p["iters"] for p in phases)

    # resume
    start_iter = 0
    resume_run_id = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt["global_iter"])
        resume_run_id = ckpt.get("run_id")
        print(f"resumed from {args.resume} at iter {start_iter}")

    logger = RunLogger(cfg.run_dir, run_id=resume_run_id)
    if not args.resume:
        logger.write_config({
            **asdict(cfg),
            "run_id": logger.run_id, "git_commit": git_commit(), "device": dev,
            "started": utc_now(), "phases": phases, "total_iters": total,
        })
    print(f"run {logger.run_id} | device={dev} | curriculum={args.curriculum} "
          f"| total_iters={total} | B={cfg.ppo.batch_games} | start={start_iter}")

    cur_phase_idx = -1
    stage = None
    env_steps = 0

    for gi in range(start_iter + 1, total + 1):
        phase, pidx = phase_at(phases, gi)
        if pidx != cur_phase_idx:
            cur_phase_idx = pidx
            stage = Stage(wl, PAT, phase["answers"], pool_size(phase["pool"]), dev, seed=cfg.seed)
            psz = pool_size(phase["pool"]) or wl.n_allowed
            print(f"\n== phase {phase['name']} | answers={phase['answers']} pool={psz} ==")

        frac = gi / total
        scoef = anneal(cfg.reward.shaping_coef, frac, cfg.reward.anneal_shaping)

        model.train()
        t0 = time.time()
        batch, rstats = collect_rollout(model, env, stage, cfg.reward, cfg.ppo.gamma,
                                        cfg.ppo.gae_lambda, scoef, dev, rng, cfg.ppo.batch_games)
        pstats = ppo_update(model, opt, batch, cfg.ppo)
        env_steps += rstats["n_transitions"]
        del batch
        if dev == "mps":
            torch.mps.empty_cache()      # prevent MPS allocator growth over a long run

        logger.log_metrics({
            "iter": gi, "phase": phase["name"], "wall_clock": utc_now(),
            "env_steps": env_steps, "games": cfg.ppo.batch_games,
            "n_answers": phase["answers"], "pool": pool_size(phase["pool"]) or wl.n_allowed,
            "train": {k: rstats[k] for k in ("win_rate", "avg_guesses", "mean_return", "mean_reward")},
            "ppo": {k: pstats[k] for k in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                            "clip_frac", "explained_variance", "grad_norm")},
            "hparams": {"lr": cfg.ppo.lr, "entropy_coef": cfg.ppo.entropy_coef, "shaping_coef": scoef},
            "iter_seconds": round(time.time() - t0, 3),
        })
        if gi == start_iter + 1 or gi % 10 == 0:
            print(f"[{phase['name']}] it {gi:5d}/{total} | win {rstats['win_rate']*100:5.1f}% "
                  f"| ret {rstats['mean_return']:6.2f} | ent {pstats['entropy']:.2f} "
                  f"| kl {pstats['approx_kl']:.3f} | ev {pstats['explained_variance']:.2f} "
                  f"| {time.time()-t0:.2f}s")

        if gi % cfg.eval_every == 0 or gi == total:
            model.eval()
            agent = ModelAgent(model, dev, pool_mask=stage.pool_mask, greedy=True)
            res = evaluate(agent, env, stage.subset_allowed_ids)
            logger.log_eval({
                "iter": gi, "phase": phase["name"], "wall_clock": utc_now(),
                "n_answers": int(stage.A), "win_rate": res.win_rate, "avg_guesses": res.avg_guesses,
                "guess_distribution": res.distribution, "failed_words": res.failed_words[:50],
            })
            print(f"   [eval {phase['name']}] iter {gi}: greedy win {res.win_rate*100:.1f}% "
                  f"on {stage.A} answers (avg {res.avg_guesses:.2f})")

        if gi % cfg.ckpt_every == 0 or gi == total:
            payload = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                       "global_iter": gi, "run_id": logger.run_id, "phase": phase["name"]}
            torch.save(payload, logger.dir / "checkpoints" / f"iter_{gi}.pt")
            torch.save(payload, logger.dir / "checkpoints" / "latest.pt")

    logger.close()
    print("done ->", logger.dir)


if __name__ == "__main__":
    main()
