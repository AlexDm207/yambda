"""Relevance + reward + behaviour of top-10 lists on val; optional RM-reranking of the top-N (alphas)."""
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
from src.rl_lib import (AC, REWARDS, combine, context_arrays, load_policy, load_rm,
                        rm_score_candidates, token_stats)
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--rm_b", default="checkpoints/rm_b/best.pt")
ap.add_argument("--reward", choices=REWARDS, default="ratio")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--n_eval", type=int, default=6000)
ap.add_argument("--alphas", type=float, nargs="*", default=[0.0], help="score = log pi + alpha * reward_A")
ap.add_argument("--n_cand", type=int, default=10)
ap.add_argument("--tag", default="")
args = ap.parse_args()
if any(a > 0 for a in args.alphas):
    args.n_cand = max(args.n_cand, 100)

dev = "cuda"
cfg = load_cfg()
thr, L, V = cfg["data"]["positive_ratio_threshold"], cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
C, K, TOP = L - 1, args.n_cand, 10        # 127 real context events + 1 candidate = 128 positions
ev = load_events()
idx, cs = eval_samples(ev, args.split, 10, seed=2024, L=L)
idx = idx[(idx - cs) >= C]                                         # full contexts only
if len(idx) > args.n_eval:                                         # the same subset for every model
    idx = np.sort(np.random.default_rng(0).choice(idx, args.n_eval, replace=False))
tok_ctx, fb_in, fb_last = context_arrays(ev, idx, C)
tgt = ev["token"][idx]
N = len(idx)
print(f"{args.split}: {N} contexts, in-vocabulary targets: {(tgt > 1).mean():.1%}")

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

P = {}
for key, path in [("A", args.rm_a), ("B", args.rm_b)]:
    rm = load_rm(path)
    P[key] = rm_score_candidates(rm, tok_ctx, fb_in, fb_last, cands, dev)
    del rm
len_sec, len_bias = token_stats(ev, V, args.reward)
R = {k: combine(P[k], cands, args.reward, len_sec, len_bias) for k in P}

rows, full = [], {}
for alpha in args.alphas:
    order = np.argsort(-(logp + alpha * R["A"]), axis=1)[:, :TOP]
    final = np.take_along_axis(cands, order, 1)
    hit = final == tgt[:, None]
    ranks = np.where(hit.any(1), hit.argmax(1) + 1, 10 ** 9)
    rel, _ = report(ranks, ev, idx, thr, ks=(10,))
    pa = np.take_along_axis(P["A"], order[..., None], 1).reshape(-1, 4).mean(0)
    flat = final.ravel()
    cnt = np.bincount(flat, minlength=V)[2:]
    srt = np.sort(cnt)
    n = len(srt)
    d = {
        "alpha": alpha,
        "R@10": rel["all"]["recall@10"], "NDCG@10": rel["all"]["ndcg@10"], "MAP@10": rel["all"]["map@10"],
        "R@10_new": rel["new"]["recall@10"], "R@10_repeat": rel["repeat"]["recall@10"],
        "R@10_pos": rel["positive"]["recall@10"],
        "reward_A": float(np.take_along_axis(R["A"], order, 1).mean()),
        "reward_B": float(np.take_along_axis(R["B"], order, 1).mean()),
        "pA_ratio": float(pa[0]), "pA_comp": float(pa[1]), "pA_like": float(pa[2]),
        "len_sec": float(len_sec[final].mean()),
        "logrank": float(np.log10(final - 1).mean()),
        "coverage": float((cnt > 0).mean()),
        "gini": float(2 * (np.arange(1, n + 1) * srt).sum() / (n * srt.sum()) - (n + 1) / n),
        "top100_share": float((flat <= 101).mean()),
        "max_item_share": float(cnt.max() / flat.size),
        "in_context_share": float((final[:, :, None] == tok_ctx[:, None, :]).any(-1).mean()),
    }
    rows.append(d)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"\nreward preset: {args.reward}; N={N}; SE of R@10 is about 0.007 (differences below ~0.015 are noise)")
print(df.round(4).to_string(index=False))
name = f"{Path(args.ckpt).parent.name}_{Path(args.ckpt).stem}"
json.dump(rows, open(ROOT / "reports/results" / f"{args.split}_reward_{name}_{args.reward}{args.tag}.json", "w"), indent=2)