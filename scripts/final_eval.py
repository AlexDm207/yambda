"""Final evaluation: per-context arrays, user-cluster bootstrap, matched-reward gap to the rerank frontier."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.loader import eval_samples, load_events, long_targets
from src.rl_lib import AC, context_arrays, load_policy, load_rm, rm_score_candidates, token_stats
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--base", default="checkpoints/base_all/best.pt")
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--rm_b", default="checkpoints/rm_b/best.pt")
ap.add_argument("--groups", nargs="*", default=[], help="name=ckpt1,ckpt2,... (one checkpoint per seed)")
ap.add_argument("--alphas", type=float, nargs="*", default=[0, 2, 5, 10, 20, 30, 50, 100, 200, 300])
ap.add_argument("--n_eval", type=int, default=6000)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--tag", default="main")
args = ap.parse_args()

dev = "cuda"
cfg = load_cfg()
L, V = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
C, TOP = L - 1, 10
ev = long_targets(load_events())
idx, cs = eval_samples(ev, args.split, 10, seed=2024, L=L)
idx = idx[(idx - cs) >= C]
if len(idx) > args.n_eval:                                   # same subset as in eval_reward.py
    idx = np.sort(np.random.default_rng(0).choice(idx, args.n_eval, replace=False))
tok_ctx, fb_in, fb_last = context_arrays(ev, idx, C)
tgt = ev["token"][idx]

real_ratio = ev["ratio"][idx].astype(np.float64) / 100.0     # real outcomes of the logged next listen
real_g3, real_g10 = ev["g"][idx, 1].astype(np.float64), ev["g"][idx, 2].astype(np.float64)
inv = (tgt > 1).astype(np.float64)                           # relevance metrics: in-vocabulary targets only
N = len(idx)
user = np.unique(np.searchsorted(ev["u_start"], idx, side="right") - 1, return_inverse=True)[1]
U = int(user.max()) + 1
len_sec, _ = token_stats(ev, V)
rm_a, rm_b = load_rm(args.rm_a), load_rm(args.rm_b)
print(f"{args.split}: {N} contexts from {U} users; in-vocabulary targets {inv.mean():.1%}")


def policy_topk(path, K):
    policy, _ = load_policy(path)
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
    torch.cuda.empty_cache()
    return cands, logp


def score(cands):
    return (rm_score_candidates(rm_a, tok_ctx, fb_in, fb_last, cands, dev),
            rm_score_candidates(rm_b, tok_ctx, fb_in, fb_last, cands, dev))


def feats(final, PA, PB):
    eq = final == tgt[:, None]
    hit = eq.any(1).astype(np.float64)
    pos = eq.argmax(1) + 1
    return {"final": final, "hit": hit, "ndcg": hit / np.log2(pos + 1), "map": hit / pos,
            "hit_ratio": hit * real_ratio, "hit_pos": hit * (real_ratio >= 0.5),
            "hit_g3": hit * real_g3, "hit_g10": hit * real_g10,
            "rA": PA[..., 0].mean(1), "rB": PB[..., 0].mean(1), "like": PA[..., 2].mean(1),
            "len": len_sec[final].mean(1), "top100": (final <= 101).mean(1),
            "ctx": (final[:, :, None] == tok_ctx[:, None, :]).any(-1).mean(1)}


members = {}                                                  # name -> list of per-seed feature dicts
cands, logp = policy_topk(args.base, 100)
PA, PB = score(cands)
for a in args.alphas:                                         # rerank frontier of the base model
    order = np.argsort(-(logp + a * PA[..., 0]), axis=1)[:, :TOP]
    fin = np.take_along_axis(cands, order, 1)
    members[f"rerank_a{a:g}"] = [feats(fin, np.take_along_axis(PA, order[..., None], 1),
                                       np.take_along_axis(PB, order[..., None], 1))]
for g in args.groups:
    name, paths = g.split("=")
    members[name] = []
    for p in paths.split(","):
        fin, _ = policy_topk(p, TOP)
        members[name].append(feats(fin, *score(fin)))
        print("evaluated", p)

REL, OTH = ["hit", "ndcg", "map", "hit_ratio", "hit_pos", "hit_g3", "hit_g10"], ["rA", "rB", "like", "len", "top100", "ctx"]
KEYS = REL + ["rA", "rB", "like", "len", "top100", "coverage", "maxshare", "rph", "g3ph"]


def metrics(f, w):
    wi = w * inv
    d = {k: float((wi * f[k]).sum() / wi.sum()) for k in REL}
    d["rph"] = d["hit_ratio"] / max(d["hit"], 1e-12)      # real played ratio per hit
    d["g3ph"] = d["hit_g3"] / max(d["hit"], 1e-12)        # real P(>=3 more full listens in session) per hit
    d.update({k: float((w * f[k]).sum() / w.sum()) for k in OTH})
    cnt = np.bincount(f["final"].ravel(), weights=np.repeat(w, f["final"].shape[1]), minlength=V)[2:]
    d["coverage"], d["maxshare"] = float((cnt > 0).mean()), float(cnt.max() / cnt.sum())
    return d


def evaluate(w):
    return {k: [metrics(f, w) for f in v] for k, v in members.items()}


def gap(res, name, axis):
    pts = sorted((res[f"rerank_a{a:g}"][0][axis], res[f"rerank_a{a:g}"][0]["hit"]) for a in args.alphas)
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    g = [m["hit"] - np.interp(m[axis], xs, ys) if xs[0] <= m[axis] <= xs[-1] else np.nan for m in res[name]]
    return float(np.mean(g))


point = evaluate(np.ones(N))
rng = np.random.default_rng(0)
boot = [evaluate(np.bincount(rng.integers(0, U, U), minlength=U)[user].astype(np.float64)) for _ in range(args.boot)]

out, rows = {}, []
for k in members:
    out[k] = {}
    for key in KEYS:
        pt = np.mean([m[key] for m in point[k]])
        bs = np.array([np.mean([m[key] for m in b[k]]) for b in boot])
        out[k][key] = [float(pt), *map(float, np.percentile(bs, [2.5, 97.5]))]
    if not k.startswith("rerank_"):
        for axis in ("rA", "rB"):
            pt, bs = gap(point, k, axis), np.array([gap(b, k, axis) for b in boot])
            ok = ~np.isnan(bs)
            out[k]["gap_" + axis] = [pt, *map(float, np.percentile(bs[ok], [2.5, 97.5]))] if ok.sum() > 50 else [pt, np.nan, np.nan]
        out[k]["seeds_R10"] = [m["hit"] for m in point[k]]


def fmt(v, d=3):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f},{v[2]:.{d}f}]"


for k in members:
    r = {"method": k, "seeds": len(members[k]), "R@10": fmt(out[k]["hit"]), "NDCG@10": fmt(out[k]["ndcg"]),
         "hit_ratio": fmt(out[k]["hit_ratio"]), "ratio/hit": fmt(out[k]["rph"]), "g>=3/hit": fmt(out[k]["g3ph"]),
         "rewardA": f"{out[k]['rA'][0]:.4f}", "rewardB": f"{out[k]['rB'][0]:.4f}", "len": f"{out[k]['len'][0]:.0f}",
         "top100": f"{out[k]['top100'][0]:.3f}", "maxshare": f"{out[k]['maxshare'][0]:.4f}",
         "coverage": f"{out[k]['coverage'][0]:.3f}"}
    if "gap_rA" in out[k]:
        r["gapA"], r["gapB"] = fmt(out[k]["gap_rA"]), fmt(out[k]["gap_rB"])
        r["R@10 by seed"] = " ".join(f"{x:.3f}" for x in out[k]["seeds_R10"])
    rows.append(r)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
print("\n95% intervals: bootstrap over users. gap = R@10 minus the R@10 of base-model reranking at the same predicted reward")
print(pd.DataFrame(rows).fillna("").to_string(index=False))
json.dump(out, open(ROOT / "reports/results" / f"{args.split}_final_{args.tag}.json", "w"), indent=2)