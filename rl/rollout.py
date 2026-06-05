"""Batched self-play rollout: collect transitions (turns 2-6) and compute GAE advantages.

The model plays turns 2-6 (opener auto-played by the env). Reward is computed here from
env state + candidate tracking (the env keeps a zero placeholder). Every episode ends by
the 5th model step (turn 6 forces done), so GAE bootstraps from 0 at the terminal step.
"""
from __future__ import annotations

import numpy as np
import torch

from model.encoder import encode
from rl.reward import CandidateTracker, observed_pattern, step_reward


def collect_rollout(model, env, stage, reward_cfg, gamma, gae_lambda,
                    shaping_coef, device, rng, B, max_steps=5):
    secrets, subcol = stage.sample_secrets(B, rng)
    obs = env.reset(secrets)

    tracker = CandidateTracker(stage.pat_sub).reset(B)
    opener_actions = np.full(B, env.opener_id, dtype=np.int64)
    tracker.update(opener_actions, observed_pattern(obs.feedbacks[:, 0, :]))
    prev_counts = tracker.counts()

    pool_mask = stage.pool_mask
    buf = {k: [] for k in ("l", "f", "t", "c", "mask", "act", "logp", "val", "rew", "done", "active")}
    prev_done = obs.done.copy()

    for _ in range(max_steps):
        if obs.done.all():
            break
        tokens, mask = encode(obs, device)
        action, logp, value, _ = model.act(tokens, mask, pool_mask=pool_mask, greedy=False)
        a_np = action.cpu().numpy()
        active = ~prev_done

        obs, _, _, _ = env.step(a_np)
        patterns = stage.pat_sub[a_np, subcol]                  # observed pattern of this guess
        tracker.update(a_np, patterns, active=active)
        new_counts = tracker.counts()
        reward = step_reward(obs.won, obs.turn, prev_counts, new_counts,
                             reward_cfg, shaping_coef, gamma)

        buf["l"].append(tokens[0]); buf["f"].append(tokens[1])
        buf["t"].append(tokens[2]); buf["c"].append(tokens[3]); buf["mask"].append(mask)
        buf["act"].append(action); buf["logp"].append(logp); buf["val"].append(value)
        buf["rew"].append(torch.from_numpy(reward).to(device))
        buf["done"].append(torch.from_numpy(obs.done.copy()).to(device))
        buf["active"].append(torch.from_numpy(active.copy()).to(device))

        prev_counts = new_counts
        prev_done = obs.done.copy()

    T = len(buf["act"])
    st = {k: torch.stack(v) for k, v in buf.items()}            # each [T,B,...]
    rew, val, done, active = st["rew"], st["val"], st["done"], st["active"]

    # GAE (inactive steps don't propagate)
    adv = torch.zeros(T, B, device=device)
    lastgae = torch.zeros(B, device=device)
    nextval = torch.zeros(B, device=device)
    for tt in reversed(range(T)):
        a = active[tt].float()
        nonterminal = (~done[tt]).float()
        delta = rew[tt] + gamma * nextval * nonterminal - val[tt]
        gae = delta + gamma * gae_lambda * nonterminal * lastgae
        lastgae = a * gae + (1 - a) * lastgae
        adv[tt] = a * gae
        nextval = a * val[tt] + (1 - a) * nextval
    returns = adv + val

    flat = active.reshape(-1)

    def fl(x):
        return x.reshape(T * B, *x.shape[2:])[flat]

    batch = {
        "tokens": (fl(st["l"]), fl(st["f"]), fl(st["t"]), fl(st["c"])),
        "mask": fl(st["mask"]),
        "actions": fl(st["act"]),
        "logprobs": fl(st["logp"]),
        "values": fl(val),
        "advantages": fl(adv),
        "returns": fl(returns),
        "pool_mask": pool_mask,
    }

    won = obs.won
    stats = {
        "win_rate": float(won.mean()),
        "avg_guesses": float(obs.turn[won].mean()) if won.any() else 0.0,
        "mean_reward": float(rew[active].mean()) if active.any() else 0.0,
        "mean_return": float(returns[active].mean()) if active.any() else 0.0,
        "n_transitions": int(flat.sum()),
    }
    return batch, stats
