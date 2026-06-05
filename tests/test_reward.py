"""Reward: terminal values, potential-based shaping, candidate tracking."""
import numpy as np

from config import RewardConfig
from rl.reward import CandidateTracker, observed_pattern, step_reward
from wordle.feedback import feedback_batch
from wordle.feedback_table import load_pattern_table
from wordle.words import load_wordlist

wl = load_wordlist()
PAT = load_pattern_table(wl)
RC = RewardConfig()


def test_terminal_reward_values():
    won = np.array([True, True, False])
    k = np.array([2, 6, 6])
    r = step_reward(won, k, np.array([10.0, 10.0, 10.0]), np.array([1.0, 1.0, 5.0]),
                    RC, shaping_coef=0.0, gamma=1.0)
    assert np.isclose(r[0], 12.0)   # win in 2: 10 + 0.5*(6-2)
    assert np.isclose(r[1], 10.0)   # win in 6: 10
    assert np.isclose(r[2], 0.0)    # loss


def test_shaping_telescopes():
    # no win, pure shaping: coef * (gamma*Φ(new) - Φ(prev)), Φ = -log2(n)
    r = step_reward(np.array([False]), np.array([3]), np.array([100.0]), np.array([1.0]),
                    RC, shaping_coef=0.8, gamma=1.0)
    assert np.isclose(r[0], 0.8 * np.log2(100))   # Φ(1)=0, Φ(100)=-log2(100)


def test_candidate_tracker_vs_bruteforce():
    tracker = CandidateTracker(PAT).reset(1)
    g = wl.id_of("salet")
    crane_idx = wl.answers.index("crane")
    fb = feedback_batch(wl.allowed_ids[[g]], wl.answers_ids[[crane_idx]])
    tracker.update(np.array([g]), observed_pattern(fb))
    assert tracker.counts()[0] == 83                 # matches brute force
    assert tracker.cand[0, crane_idx]                # true secret stays consistent
