"""Configuration dataclasses for M1 training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pool_size(p):
    """CLI convention: -1 (or None) means the full vocabulary (no guess-pool restriction)."""
    return None if (p is not None and p < 0) else p


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0


@dataclass
class PPOConfig:
    batch_games: int = 2048
    epochs: int = 4
    minibatch: int = 4096
    lr: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    kl_target: float = 0.02
    grad_clip: float = 0.5


@dataclass
class RewardConfig:
    win_base: float = 10.0
    win_speed: float = 0.5
    loss: float = 0.0
    shaping_coef: float = 0.8
    anneal_shaping: bool = True


@dataclass
class CurriculumConfig:
    stage: str = "A"
    n_answers: int = 200
    guess_pool_size: int | None = 800   # None => full 12,972
    opener: str = "salet"
    max_guesses: int = 6


@dataclass
class RunConfig:
    seed: int = 0
    total_iters: int = 300
    eval_every: int = 20
    ckpt_every: int = 50
    games_sample: int = 16
    run_dir: Path = REPO_ROOT / "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
