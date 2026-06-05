"""Model: shapes, param count, near-uniform init, act/evaluate_actions parity."""
import math

import torch
from torch.distributions import Categorical

from config import ModelConfig
from model.encoder import SEQ_LEN
from model.transformer import N_WORDS, WordleTransformer
from wordle.words import load_wordlist

wl = load_wordlist()


def _toks(B, valid=True):
    letter = torch.randint(0, 28, (B, SEQ_LEN))
    fb = torch.randint(0, 4, (B, SEQ_LEN))
    turn = torch.randint(0, 6, (B, SEQ_LEN))
    col = torch.randint(0, 5, (B, SEQ_LEN))
    mask = torch.zeros(B, SEQ_LEN, dtype=torch.bool)   # all valid
    return (letter, fb, turn, col), mask


def test_param_count_around_6_5M():
    m = WordleTransformer(ModelConfig(), wl.allowed_ids)
    n = sum(p.numel() for p in m.parameters())
    assert 2_500_000 < n < 4_500_000, n


def test_forward_shapes():
    m = WordleTransformer(ModelConfig(), wl.allowed_ids)
    toks, mask = _toks(4)
    logits, value = m(toks, mask)
    assert logits.shape == (4, N_WORDS)
    assert value.shape == (4,)


def test_initial_policy_near_uniform():
    torch.manual_seed(0)
    m = WordleTransformer(ModelConfig(), wl.allowed_ids)
    toks, mask = _toks(8)
    logits, _ = m(toks, mask)
    ent = Categorical(logits=logits).entropy().mean().item()
    assert ent > math.log(N_WORDS) - 0.5, ent      # ~9.47 nats => broad exploration


def test_act_evaluate_actions_parity():
    torch.manual_seed(0)
    m = WordleTransformer(ModelConfig(), wl.allowed_ids)
    toks, mask = _toks(4)
    a, lp, v, ent = m.act(toks, mask)
    lp2, ent2, v2 = m.evaluate_actions(toks, mask, a)
    assert torch.allclose(lp, lp2, atol=1e-4)
    assert torch.allclose(v, v2, atol=1e-4)
    assert torch.allclose(ent, ent2, atol=1e-4)
