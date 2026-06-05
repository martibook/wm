"""Rollout buffer + GAE shapes, done-masking, and one finite PPO step that changes params."""
import numpy as np
import torch

from config import ModelConfig, PPOConfig, RewardConfig
from model.transformer import WordleTransformer
from rl.curriculum import Stage
from rl.ppo import ppo_update
from rl.rollout import collect_rollout
from wordle.env import WordleEnv
from wordle.feedback_table import load_pattern_table
from wordle.words import load_wordlist

wl = load_wordlist()
PAT = load_pattern_table(wl)
DEV = "cpu"


def _setup(B=32, n_answers=20, pool=60):
    torch.manual_seed(0)
    model = WordleTransformer(ModelConfig(), wl.allowed_ids).to(DEV)
    env = WordleEnv(wl, opener="salet")
    stage = Stage(wl, PAT, n_answers=n_answers, guess_pool_size=pool, device=DEV, seed=0)
    return model, env, stage


def test_rollout_shapes_and_done_masking():
    model, env, stage = _setup()
    rng = np.random.default_rng(0)
    batch, stats = collect_rollout(model, env, stage, RewardConfig(), gamma=0.99,
                                   gae_lambda=0.95, shaping_coef=0.8, device=DEV, rng=rng, B=32)
    n = batch["actions"].shape[0]
    assert n == stats["n_transitions"] and n > 0
    assert batch["tokens"][0].shape == (n, 26)
    assert batch["advantages"].shape == (n,) and torch.isfinite(batch["advantages"]).all()
    assert batch["returns"].shape == (n,) and torch.isfinite(batch["returns"]).all()
    # each game makes between 1 and 5 model guesses -> transitions in [B, 5B]
    assert 32 <= n <= 32 * 5
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_ppo_step_changes_params_and_is_finite():
    model, env, stage = _setup()
    rng = np.random.default_rng(1)
    batch, _ = collect_rollout(model, env, stage, RewardConfig(), gamma=0.99,
                               gae_lambda=0.95, shaping_coef=0.8, device=DEV, rng=rng, B=64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = model.policy_head.proj.weight.detach().clone()
    cfg = PPOConfig(epochs=2, minibatch=128)
    stats = ppo_update(model, opt, batch, cfg)
    assert all(np.isfinite(stats[k]) for k in ("policy_loss", "value_loss", "entropy", "approx_kl"))
    assert not torch.allclose(before, model.policy_head.proj.weight)   # params moved
