"""Multi-step diagnostics: free-running (own top-1 items fed back) vs teacher-forced (real items)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.loader import load_events
from src.rl_lib import AC, context_arrays, load_policy, load_rm, token_stats
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--H", type=int, default=5)
ap.add_argument("--per_user", type=int, default=5)
ap.add_argument("--n_eval", type=int, default=4000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--no_repeat", action="store_true")
args = ap.parse_args()

dev = "cuda"
cfg = load_cfg()
L, V = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
H = args.H
C = L - H
ev = load_events()
len_sec, _ = token_stats(ev, V)

# contexts with C real events before and H real events after, all inside the evaluation period
rng = np.random.default_rng(2024)
a, b = ev["bounds"][args.split]
idx = []
for us, lo0, hi0 in zip(ev["u_start"], a, b):
    lo, hi = max(lo0, us + C), hi0 - H
    if hi >= lo:
        idx.append(rng.choice(np.arange(lo, hi + 1), size=min(args.per_user, hi - lo + 1), replace=False))
idx = np.concatenate(idx)
if len(idx) > args.n_eval:
    idx = np.sort(rng.choice(idx, args.n_eval, replace=False))
N = len(idx)

tok, fb_in, fb_last = context_arrays(ev, idx, C)
pos = idx[:, None] + np.arange(H)[None, :]
fut_tok = ev["token"][pos]
fr = ev["ratio"][pos].astype(np.float32) / 100.0
fut_fb = np.stack([fr, (fr >= 0.5).astype(np.float32),
                   ev["like"][pos].astype(np.float32), ev["dislike"][pos].astype(np.float32)], -1)
valid = fut_tok > 1
print(f"{args.split}: {N} contexts, C={C}, H={H}; in-vocabulary real targets: {valid.mean():.1%}")

policy, _ = load_policy(args.ckpt)
policy.eval()
rm = load_rm(args.rm_a)
W = policy.tok.weight[2:].float()


def rank_of(lg, t):
    ok = (t > 1).cpu().numpy()
    tl = lg.gather(1, (t.clamp(min=2) - 2).unsqueeze(1))
    r = ((lg > tl).sum(1) + 1).cpu().numpy()
    return np.where(ok, r, 10 ** 9)


@torch.no_grad()
def rm_pred(seq, fb_c, fb_l, mode, real_fb=None):
    """RM probabilities for the H items after the context; feedback of earlier items is set according to `mode`."""
    n = seq.shape[0]
    fb = torch.zeros(n, C + H, 4, device=dev)
    fb[:, :C] = fb_c
    fb[:, C] = fb_l
    outs = []
    for t in range(H):
        with torch.autocast(**AC):
            out = rm(seq, fb)
        p = torch.sigmoid(out[:, C + t].float())
        outs.append(p)
        if t + 1 < H:
            if mode == "sim":
                fb[:, C + t + 1] = p
            elif mode == "pos":
                fb[:, C + t + 1] = torch.tensor([1.0, 1.0, 0.0, 0.0], device=dev)
            elif mode == "real":
                assert real_fb is not None, "mode='real' requires real_fb"
                fb[:, C + t + 1] = real_fb[:, t]
    return torch.stack(outs, 1).cpu().numpy()


tf_rank = np.full((N, H), 10 ** 9, dtype=np.int64)
fr_rank = tf_rank.copy()
co_rank = tf_rank.copy()
gen = np.zeros((N, H), dtype=np.int64)
P = {k: np.zeros((N, H, 4), np.float32) for k in ["real", "sim", "zero", "pos"]}

for s in range(0, N, args.bs):
    sl = slice(s, min(s + args.bs, N))
    ctx = torch.from_numpy(tok[sl]).long().to(dev)
    fut = torch.from_numpy(fut_tok[sl]).long().to(dev)
    fb_c = torch.from_numpy(fb_in[sl]).to(dev)
    fb_l = torch.from_numpy(fb_last[sl]).to(dev)
    rfb = torch.from_numpy(fut_fb[sl]).to(dev)
    with torch.no_grad():
        with torch.autocast(**AC):                                   # teacher forcing: one pass over real items
            h = policy.hidden(torch.cat([ctx, fut], 1)[:, :-1])[:, C - 1:]
        lg = h.float() @ W.T
        for j in range(H):
            tf_rank[sl, j] = rank_of(lg[:, j], fut[:, j])
        del lg
        x = ctx                                                       # free-running: own top-1 items are fed back
        for j in range(H):
            with torch.autocast(**AC):
                hj = policy.hidden(x)[:, -1]
            lgj = hj.float() @ W.T
            if j == 0:                                    # control: context only, no generated items
                for jj in range(H):
                    co_rank[sl, jj] = rank_of(lgj, fut[:, jj])
            if args.no_repeat and j > 0:                  # slate mode: never recommend an item twice
                lgj.scatter_(1, x[:, C:] - 2, float("-inf"))
            fr_rank[sl, j] = rank_of(lgj, fut[:, j])
            nxt = lgj.argmax(-1) + 2
            gen[sl, j] = nxt.cpu().numpy()
            x = torch.cat([x, nxt[:, None]], 1)
    for mode in ["sim", "zero", "pos"]:
        P[mode][sl] = rm_pred(x, fb_c, fb_l, mode)
    P["real"][sl] = rm_pred(torch.cat([ctx, fut], 1), fb_c, fb_l, "real", rfb)

rows = []
for j in range(H):
    v, g = valid[:, j], gen[:, j]
    rows.append({
        "step": j + 1,
        "R10_teacher": float((tf_rank[v, j] <= 10).mean()),
        "R10_free": float((fr_rank[v, j] <= 10).mean()),
        "R10_ctxonly": float((co_rank[v, j] <= 10).mean()),
        f"hit_real_next{H}": float((g[:, None] == fut_tok).any(1).mean()),
        "dup_in_seq": float((g[:, None] == gen[:, :j]).any(1).mean()) if j else 0.0,
        "in_context": float((g[:, None] == tok).any(1).mean()),
        "logrank": float(np.log10(g - 1).mean()),
        "len_sec": float(len_sec[g].mean()),
        "uniq_items": float(len(np.unique(g)) / N),
        "rm_real": float(P["real"][:, j, 0][v].mean()),
        "actual_real": float(fr[:, j][v].mean()),
        "rm_sim": float(P["sim"][:, j, 0].mean()),
        "rm_zero": float(P["zero"][:, j, 0].mean()),
        "rm_pos": float(P["pos"][:, j, 0].mean()),
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"\n{Path(args.ckpt).parent.name}/{Path(args.ckpt).stem}: greedy top-1 feedback; SE of R@10 about 0.008")
print(df.round(4).to_string(index=False))
json.dump(rows, open(ROOT / "reports/results" / f"{args.split}_multistep_{Path(args.ckpt).parent.name}_{Path(args.ckpt).stem}{'_norepeat' if args.no_repeat else ''}.json", "w"), indent=2)