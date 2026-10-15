from typing import Any

import numpy as np
import torch

from models.gpt_rec import GPTRec, GPTRecConfig
from models.reward_model import RewardModel
from src.utils import ROOT

REWARDS = ["ratio", "comp", "seconds", "composite", "debiased", "itemdebiased"]
AC: dict[str, Any] = dict(device_type="cuda", dtype=torch.bfloat16)


def load_policy(path, device="cuda"):
    ck = torch.load(ROOT / path, map_location="cpu")
    m = GPTRec(GPTRecConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(device), ck["cfg"]


def load_rm(path, device="cuda"):
    ck = torch.load(ROOT / path, map_location="cpu")
    c = GPTRecConfig(**ck["cfg"])
    m = RewardModel(c, torch.zeros(c.vocab_size))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def token_stats(ev, V, reward="ratio"):
    """Per-token mean track length (sec) and length-bin mean ratio, from TRAIN events only."""
    m = ev["period"] == 0
    tok = ev["token"][m]
    ln = ev["length"][m].astype(np.float64)
    r = ev["ratio"][m].astype(np.float64) / 100.0
    cnt = np.bincount(tok, minlength=V)
    len_sec = np.where(cnt > 0, np.bincount(tok, weights=ln, minlength=V) / np.maximum(cnt, 1), ln.mean())
    b = np.minimum((ln // 60).astype(np.int64), 8)
    bin_mean = np.bincount(b, weights=r, minlength=9) / np.maximum(np.bincount(b, minlength=9), 1)
    len_bias = bin_mean[np.minimum((len_sec // 60).astype(np.int64), 8)]
    if reward == "itemdebiased":
        g = r.mean()
        item_mean = (np.bincount(tok, weights=r, minlength=V) + 20 * g) / (cnt + 20)   # shrunk item mean, train only
        return len_sec.astype(np.float32), item_mean.astype(np.float32)
    return len_sec.astype(np.float32), len_bias.astype(np.float32)


def combine(P, tok, preset, len_sec, len_bias):
    """P[..., 4] = probabilities [ratio, completed, like, dislike]; works for numpy and torch."""
    pr, pc, pl, pd = P[..., 0], P[..., 1], P[..., 2], P[..., 3]
    if preset == "ratio":
        return pr
    if preset == "comp":
        return pc
    if preset == "seconds":
        return pr * len_sec[tok] / 200.0
    if preset == "composite":
        return pc + 10.0 * pl - 5.0 * pd
    if preset in ("debiased", "itemdebiased"):
        return pr - len_bias[tok]
    raise ValueError(preset)


def context_arrays(ev, idx, C):
    """Context = events [idx-C, idx). Returns tokens, shifted feedback inputs, feedback of the last context event."""
    pos = (idx - C)[:, None] + np.arange(C)[None, :]
    tok = ev["token"][pos]
    r = ev["ratio"][pos].astype(np.float32) / 100.0
    fb = np.stack([r, (r >= 0.5).astype(np.float32),
                   ev["like"][pos].astype(np.float32), ev["dislike"][pos].astype(np.float32)], -1)
    fb_in = np.concatenate([np.zeros_like(fb[:, :1]), fb[:, :-1]], 1)
    return tok, fb_in, fb[:, -1]


@torch.no_grad()
def rollout(policy, ctx, H):
    """Sample H items autoregressively (temperature 1, full in-vocabulary softmax)."""
    x = ctx
    W = policy.tok.weight[2:].float()
    for _ in range(H):
        with torch.autocast(**AC):
            h = policy.hidden(x)[:, -1]
        p = torch.softmax(h.float() @ W.T, dim=-1)
        x = torch.cat([x, torch.multinomial(p, 1) + 2], 1)
    return x


@torch.no_grad()
def rm_rollout(rm, x, fb_ctx, fb_last, C, H):
    """Rewards for generated items. Feedback of generated items is SIMULATED by the RM's own probabilities."""
    M, T = x.shape
    fb = torch.zeros(M, T, 4, device=x.device)
    fb[:, :C] = fb_ctx
    fb[:, C] = fb_last                      # real feedback of the last real event
    P = []
    for t in range(H):
        with torch.autocast(**AC):
            out = rm(x, fb)
        p = torch.sigmoid(out[:, C + t].float())
        P.append(p)
        if C + t + 1 < T:
            fb[:, C + t + 1] = p
    return torch.stack(P, 1)                # (M, H, 4)


@torch.no_grad()
def rm_score_candidates(rm, ctx_tok, ctx_fb_in, fb_last, cands, device="cuda"):
    """One-step RM probabilities for K candidates after each real context. Returns (N, K, 4)."""
    N, K = cands.shape
    C = ctx_tok.shape[1]
    chunk = max(1, 1024 // K)
    out_all = np.zeros((N, K, 4), np.float32)
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        n = len(ctx_tok[sl])
        x = torch.cat([torch.from_numpy(ctx_tok[sl]).long().to(device).repeat_interleave(K, 0),
                       torch.from_numpy(cands[sl]).long().to(device).reshape(-1, 1)], 1)
        fb = torch.zeros(n * K, C + 1, 4, device=device)
        fb[:, :C] = torch.from_numpy(ctx_fb_in[sl]).to(device).repeat_interleave(K, 0)
        fb[:, C] = torch.from_numpy(fb_last[sl]).to(device).repeat_interleave(K, 0)
        with torch.autocast(**AC):
            out = rm(x, fb)
        out_all[sl] = torch.sigmoid(out[:, C].float()).view(n, K, 4).cpu().numpy()
    return out_all