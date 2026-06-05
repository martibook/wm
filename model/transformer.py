"""Char-level transformer: pre-LN trunk + D1 word-policy head + value head.

Reads the four interleaved token-id channels (see encoder.py), sums their embeddings,
runs a small pre-LN transformer, and reads the CLS position for the heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

N_LETTER = 28      # a..z + CLS + PAD
N_FEEDBACK = 4     # gray/yellow/green + none
N_TURN = 6         # turns 0..5
N_COL = 5          # cols 0..4
N_WORDS = 12972    # action space (allowed guesses)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.w2(self.act(self.w1(self.ln2(x))))
        return x


class WordleTransformer(nn.Module):
    def __init__(self, cfg, n_words: int = N_WORDS):
        super().__init__()
        d = cfg.d_model
        self.n_layers = cfg.n_layers
        self.letter_emb = nn.Embedding(N_LETTER, d)
        self.fb_emb = nn.Embedding(N_FEEDBACK, d)
        self.turn_emb = nn.Embedding(N_TURN, d)
        self.col_emb = nn.Embedding(N_COL, d)
        self.blocks = nn.ModuleList(
            [Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, n_words)
        self.value_head = nn.Linear(d, 1)

        self.apply(self._init_weights)
        self._scale_residual_projections()
        self._init_heads()

    # ---- initialization (GPT-2/nanoGPT style + RL head twist) ----
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _scale_residual_projections(self):
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for blk in self.blocks:
            blk.attn.out_proj.weight.data.mul_(scale)
            blk.w2.weight.data.mul_(scale)

    def _init_heads(self):
        nn.init.normal_(self.policy_head.weight, std=0.02 * 0.01)  # near-uniform initial policy
        nn.init.zeros_(self.policy_head.bias)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    # ---- forward / APIs ----
    def forward(self, tokens, key_padding_mask):
        letter_tok, fb_tok, turn_tok, col_tok = tokens
        x = (self.letter_emb(letter_tok) + self.fb_emb(fb_tok)
             + self.turn_emb(turn_tok) + self.col_emb(col_tok))
        for blk in self.blocks:
            x = blk(x, key_padding_mask)
        h = self.ln_f(x[:, 0])                      # CLS pooling
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @staticmethod
    def _apply_pool(logits, pool_mask):
        if pool_mask is None:
            return logits
        return logits.masked_fill(~pool_mask, float("-inf"))

    @torch.no_grad()
    def act(self, tokens, mask, pool_mask=None, greedy: bool = False):
        logits, value = self.forward(tokens, mask)
        logits = self._apply_pool(logits, pool_mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, tokens, mask, actions, pool_mask=None):
        logits, value = self.forward(tokens, mask)
        logits = self._apply_pool(logits, pool_mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
