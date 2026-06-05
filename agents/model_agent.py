"""Wrap a transformer as an Agent so it drops into eval.py / play.py unchanged."""
from __future__ import annotations

import numpy as np
import torch

from config import ModelConfig, device as default_device
from model.encoder import encode
from model.transformer import WordleTransformer


class ModelAgent:
    name = "model"

    def __init__(self, model, dev, pool_mask=None, greedy: bool = True):
        self.model = model.eval()
        self.dev = dev
        self.pool_mask = pool_mask    # [N_WORDS] bool tensor on device, or None (full action space)
        self.greedy = greedy

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        tokens, mask = encode(obs, self.dev)
        action, _, _, _ = self.model.act(tokens, mask, pool_mask=self.pool_mask, greedy=self.greedy)
        return action.cpu().numpy()

    @classmethod
    def untrained(cls, wl, dev=None, seed: int = 0, greedy: bool = True):
        dev = dev or default_device()
        torch.manual_seed(seed)
        model = WordleTransformer(ModelConfig(), wl.allowed_ids).to(dev)
        return cls(model, dev, greedy=greedy)

    @classmethod
    def from_checkpoint(cls, path, wl, dev=None, greedy: bool = True):
        dev = dev or default_device()
        model = WordleTransformer(ModelConfig(), wl.allowed_ids).to(dev)
        obj = torch.load(path, map_location=dev)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        model.load_state_dict(state)
        return cls(model, dev, greedy=greedy)
