"""Encoder: Obs -> interleaved token ids + padding mask (the env<->model bridge)."""
import numpy as np

from model.encoder import FB_NONE, LETTER_CLS, LETTER_PAD, SEQ_LEN, encode
from wordle.env import WordleEnv
from wordle.words import load_wordlist, word_to_letter_ids

wl = load_wordlist()


def test_shapes_and_opener():
    env = WordleEnv(wl, opener="salet")
    obs = env.reset(np.array([wl.id_of("crane")]))     # turn 1 (opener played)
    (lt, ft, tt, ct), mask = encode(obs, "cpu")

    assert lt.shape == (1, SEQ_LEN) and mask.shape == (1, SEQ_LEN)
    assert lt[0, 0].item() == LETTER_CLS               # CLS at pos 0
    assert ft[0, 0].item() == FB_NONE
    assert lt[0, 1:6].tolist() == word_to_letter_ids("salet").tolist()
    assert ft[0, 1:6].tolist() == [0, 1, 0, 1, 0]      # salet vs crane = BYBYB
    assert tt[0, 0].item() == 1                         # CLS carries decision turn
    assert tt[0, 1:6].tolist() == [0, 0, 0, 0, 0]       # opener is turn-row 0
    assert ct[0, 1:6].tolist() == [0, 1, 2, 3, 4]       # columns
    # valid length = 1 + 1*5 = 6; rest padded
    assert (~mask[0, :6]).all().item() and mask[0, 6:].all().item()
    assert lt[0, 6].item() == LETTER_PAD


def test_two_turns():
    env = WordleEnv(wl, opener="salet")
    obs = env.reset(np.array([wl.id_of("crane")]))
    obs, _, _, _ = env.step(np.array([wl.id_of("brace")]))   # turn 2
    (lt, ft, tt, ct), mask = encode(obs, "cpu")

    assert tt[0, 0].item() == 2
    assert lt[0, 6:11].tolist() == word_to_letter_ids("brace").tolist()
    assert ft[0, 6:11].tolist() == [0, 2, 2, 1, 2]      # brace vs crane = BGGYG
    assert tt[0, 6:11].tolist() == [1, 1, 1, 1, 1]      # second turn-row
    assert (~mask[0, :11]).all().item() and mask[0, 11:].all().item()


def test_batched():
    env = WordleEnv(wl, opener="salet")
    obs = env.reset(np.array([wl.id_of("crane"), wl.id_of("mango")]))
    (lt, ft, tt, ct), mask = encode(obs, "cpu")
    assert lt.shape == (2, SEQ_LEN)
