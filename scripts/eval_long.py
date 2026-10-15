"""Rerank base_all top-100 by an immediate reward (rm_a) or by the long-term model (rm_long); report both kinds of outcome."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.loader import eval_samples, load_events
from eval.metrics import report
from src.rl_lib import AC, context_arrays, load_policy, load_rm, rm_score_candidates
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/base_all/best.pt")
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--rm_long", default="checkpoints/rm_long/best.pt")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--alphas", type=float, nargs="*", default=[0, 5, 10, 20, 50])
ap.add_argument("--n_eval", type=int, default=6000)
ap.add_argument("--anova", action="store_true")
args = ap.parse_args()

dev = "cuda"
cfg = load_cfg()
thr, L, Vn = cfg["data"]["positive_ratio_threshold"], cfg["data"]["max_len"], cfg["data"]["vocab_size"]
C, K, TOP = L - 1, 100, 10
ev = load_events()
idx, cs = eval_samples(ev, args.split, 10, seed=2024, L=L)
idx = idx[(idx - cs) >= C]
if len(idx) > args.n_eval:
    idx = np.sort(np.random.default_rng(0).choice(idx, args.n_eval, replace=False))
tok_ctx, fb_in, fb_last = context_arrays(ev, idx, C)
tgt = ev["token"][idx]
N = len(idx)

policy, _ = load_policy(args.ckpt)
policy.eval()
cands, logp = np.zeros((N, K), np.int64), np.zeros((N, K), np.float32)
with torch.no_grad():
    W = policy.tok.weight[2:].float()
    for s in range(0, N, 256):
        with torch.autocast(**AC):
            h = policy.hidden(torch.from_numpy(tok_ctx[s:s + 256]).long().to(dev))[:, -1]
        v, i = torch.log_softmax(h.float() @ W.T, -1).topk(K, -1)
        cands[s:s + 256], logp[s:s + 256] = (i + 2).cpu().numpy(), v.cpu().numpy()
del policy

rm_a, rm_l = load_rm(args.rm_a), load_rm(args.rm_long)
PA = rm_score_candidates(rm_a, tok_ctx, fb_in, fb_last, cands, dev)
PL = rm_score_candidates(rm_l, tok_ctx, fb_in, fb_last, cands, dev)
R = {"ratio": PA[..., 0], "long": PL[..., :3].mean(-1) + PL[..., 3]}

rows = []
for mode in ["ratio", "long"]:
    for alpha in args.alphas:
        order = np.argsort(-(logp + alpha * R[mode]), axis=1)[:, :TOP]
        final = np.take_along_axis(cands, order, 1)
        hit = final == tgt[:, None]
        ranks = np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 9)
        rel, _ = report(ranks, ev, idx, thr, ks=(10,))
        pl_ = np.take_along_axis(PL, order[..., None], 1).reshape(-1, 4).mean(0)
        rows.append({
            "mode": mode, "alpha": alpha,
            "R@10": rel["all"]["recall@10"], "R@10_new": rel["new"]["recall@10"],
            "imm_ratio": float(np.take_along_axis(PA[..., 0], order, 1).mean()),
            "g>=1": float(pl_[0]), "g>=3": float(pl_[1]), "g>=10": float(pl_[2]), "next_day": float(pl_[3]),
            "top100_share": float((final <= 101).mean()),
            "coverage": float(len(np.unique(final)) / Vn),
            "in_context": float((final[:, :, None] == tok_ctx[:, None, :]).any(-1).mean()),
        })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"\nN={N}; SE of R@10 about 0.007; rows 'ratio' rerank by immediate RM, rows 'long' by the long-term model")
print(df.round(4).to_string(index=False))
json.dump(rows, open(ROOT / "reports/results" / f"{args.split}_long_rerank.json", "w"), indent=2)

if args.anova:
    rng = np.random.default_rng(0)
    sub = rng.choice(N, 600, replace=False)
    items = rng.choice(np.arange(2, 10002), 200, replace=False)
    Pq = rm_score_candidates(rm_l, tok_ctx[sub], fb_in[sub], fb_last[sub], np.tile(items, (600, 1)), dev)
    print("\nvariance decomposition of long-term predictions (random top-10k items)")
    for k, name in enumerate(["g>=1", "g>=3", "g>=10", "next-day"]):
        Y = Pq[..., k]
        g = Y.mean()
        r = Y.mean(1, keepdims=True) - g
        c = Y.mean(0, keepdims=True) - g
        res = Y - g - r - c
        tot = ((Y - g) ** 2).sum()
        print(f"{name:9s} context {(r**2).sum() * Y.shape[1] / tot:6.1%} | item {(c**2).sum() * Y.shape[0] / tot:6.1%} | interaction {(res**2).sum() / tot:6.1%}")