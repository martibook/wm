"""Encode the env's semantic `Obs` into the interleaved token tensors the model reads.

One token per board cell (letter + feedback + turn + col), plus a leading CLS token.
Fixed sequence length S = 1 + 5 turns * 5 cols = 26. The model sums four embedding
channels keyed by these token-id tensors; here we just build the integer ids + mask.

Token-id conventions:
  letter ids: 0..25 = a..z (as stored in Obs.guesses), 26 = CLS, 27 = PAD
  feedback :  0=gray 1=yellow 2=green (as in Obs.feedbacks), 3 = none (CLS/PAD)
  turn     :  0..5  (cells use 0-based turn; CLS carries the decision turn)
  col      :  0..4
"""
from __future__ import annotations

import numpy as np
import torch

LETTER_CLS = 26
LETTER_PAD = 27
FB_NONE = 3
MAX_CTX_TURNS = 5
SEQ_LEN = 1 + MAX_CTX_TURNS * 5   # 26

# Static per-position turn/col for the 25 cell positions (pos 1..25). Pos 0 = CLS.
_cell = np.arange(MAX_CTX_TURNS * 5)
_POS_TURN = np.concatenate([[0], _cell // 5]).astype(np.int64)   # pos0 overwritten by decision turn
_POS_COL = np.concatenate([[0], _cell % 5]).astype(np.int64)     # pos0 col unused (masked)


def encode(obs, device):
    """Obs -> ((letter, fb, turn, col) id tensors [B,26], key_padding_mask [B,26] bool).

    key_padding_mask follows torch convention: True = ignore (padding).
    """
    guesses = obs.guesses[:, :MAX_CTX_TURNS, :]      # [B,5,5] int8
    feedbacks = obs.feedbacks[:, :MAX_CTX_TURNS, :]  # [B,5,5]
    turn = obs.turn                                  # [B]
    B = guesses.shape[0]

    letters_cells = guesses.reshape(B, -1).astype(np.int64)    # [B,25], cell = t*5 + c
    fb_cells = feedbacks.reshape(B, -1).astype(np.int64)       # [B,25]

    letter_tok = np.full((B, SEQ_LEN), LETTER_PAD, dtype=np.int64)
    fb_tok = np.full((B, SEQ_LEN), FB_NONE, dtype=np.int64)
    letter_tok[:, 0] = LETTER_CLS
    fb_tok[:, 0] = FB_NONE
    letter_tok[:, 1:] = letters_cells
    fb_tok[:, 1:] = fb_cells

    turn_tok = np.broadcast_to(_POS_TURN, (B, SEQ_LEN)).copy()
    col_tok = np.broadcast_to(_POS_COL, (B, SEQ_LEN)).copy()
    decision_turn = np.clip(turn, 0, MAX_CTX_TURNS)
    turn_tok[:, 0] = decision_turn                              # CLS carries decision turn

    valid_len = 1 + decision_turn * 5                          # [B]
    positions = np.arange(SEQ_LEN)[None, :]
    pad_mask = positions < valid_len[:, None]                  # [B,26] True = valid
    invalid = ~pad_mask
    letter_tok[invalid] = LETTER_PAD
    fb_tok[invalid] = FB_NONE

    def to(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    tokens = (to(letter_tok), to(fb_tok), to(turn_tok), to(col_tok))
    key_padding_mask = to(~pad_mask)                           # True = ignore
    return tokens, key_padding_mask
