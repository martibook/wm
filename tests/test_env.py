"""Env behavior: opener auto-play, win/lose, turn counting, batching."""
import numpy as np

from wordle.env import WordleEnv
from wordle.words import load_wordlist

wl = load_wordlist()


def _id(w):
    return wl.id_of(w)


def test_opener_played_on_reset():
    env = WordleEnv(wl, opener="salet")
    obs = env.reset(np.array([_id("crane")]))
    assert obs.turn[0] == 1            # opener counts as turn 1
    assert not obs.done[0]
    # opener feedback recorded: salet vs crane -> B Y B Y B = [0,1,0,1,0]
    assert list(obs.feedbacks[0, 0]) == [0, 1, 0, 1, 0]


def test_win_sets_done_and_turn():
    env = WordleEnv(wl, opener="salet")
    obs = env.reset(np.array([_id("crane")]))
    obs, _, done, _ = env.step(np.array([_id("crane")]))
    assert obs.won[0] and obs.done[0] and done[0]
    assert obs.turn[0] == 2            # opener + winning guess


def test_loss_at_max_guesses():
    env = WordleEnv(wl, opener="salet", max_guesses=6)
    obs = env.reset(np.array([_id("crane")]))
    wrong = np.array([_id("abbey")])   # never equals crane
    for _ in range(5):                 # opener + 5 wrong = 6 guesses
        obs, _, _, _ = env.step(wrong)
    assert obs.turn[0] == 6
    assert obs.done[0] and not obs.won[0]


def test_done_games_ignore_further_actions():
    env = WordleEnv(wl, opener="salet")
    env.reset(np.array([_id("crane")]))
    env.step(np.array([_id("crane")]))     # win -> done at turn 2
    obs, _, _, _ = env.step(np.array([_id("abbey")]))
    assert obs.turn[0] == 2                 # unchanged after done
    assert obs.won[0]


def test_batched_independent_games():
    env = WordleEnv(wl, opener="salet")
    secrets = np.array([_id("crane"), _id("abbey"), _id("mango")])
    env.reset(secrets)
    # game 0 guesses correctly, game 1 guesses wrong, game 2 guesses correctly
    obs, _, _, _ = env.step(np.array([_id("crane"), _id("table"), _id("mango")]))
    assert list(obs.won) == [True, False, True]
    assert not env.all_done                       # game 1 still going
    assert obs.turn[1] == 2 and not obs.done[1]
