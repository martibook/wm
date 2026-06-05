"""From-scratch PPO update: clipped policy + value loss, entropy bonus, KL early-stop."""
from __future__ import annotations

import torch


def ppo_update(model, optimizer, batch, cfg):
    toks = batch["tokens"]
    mask = batch["mask"]
    actions = batch["actions"]
    old_logp = batch["logprobs"]
    old_values = batch["values"]
    returns = batch["returns"]
    adv = batch["advantages"]
    pool_mask = batch.get("pool_mask")

    N = actions.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    last = {}
    clip_fracs = []
    for _ in range(cfg.epochs):
        idx = torch.randperm(N, device=actions.device)
        epoch_kls = []
        for start in range(0, N, cfg.minibatch):
            mb = idx[start:start + cfg.minibatch]
            mtoks = tuple(t[mb] for t in toks)
            new_logp, entropy, value = model.evaluate_actions(mtoks, mask[mb], actions[mb], pool_mask=pool_mask)

            ratio = (new_logp - old_logp[mb]).exp()
            a = adv[mb]
            p_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()

            v_clipped = old_values[mb] + (value - old_values[mb]).clamp(-cfg.clip, cfg.clip)
            v_loss = 0.5 * torch.max((value - returns[mb]) ** 2, (v_clipped - returns[mb]) ** 2).mean()

            ent = entropy.mean()
            loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_logp[mb] - new_logp).mean().item()
                epoch_kls.append(approx_kl)
                clip_fracs.append(((ratio - 1.0).abs() > cfg.clip).float().mean().item())
            last = {
                "policy_loss": p_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": ent.item(),
                "grad_norm": float(grad_norm),
                "approx_kl": approx_kl,
            }
        if sum(epoch_kls) / len(epoch_kls) > 1.5 * cfg.kl_target:
            break

    with torch.no_grad():
        ev = 1.0 - (returns - old_values).var() / (returns.var() + 1e-8)
    last["explained_variance"] = float(ev)
    last["clip_frac"] = sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0
    return last
