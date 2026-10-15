import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.loader import eval_samples, load_events, long_targets, make_long_batch, train_windows
from eval.rm_eval import rm_predict
from models.gpt_rec import GPTRecConfig
from models.reward_model import RewardModel
from src.utils import ROOT, load_cfg, set_seed

cfg = load_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="rm_long")
ap.add_argument("--seed", type=int, default=cfg["seed"])
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--max_windows", type=int, default=0)
args = ap.parse_args()

set_seed(args.seed)
L, V, dev = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2, "cuda"
rng = np.random.default_rng(args.seed)
ev = long_targets(load_events())

m = ev["period"] == 0
ll = np.log1p(ev["length"][m].astype(np.float64))
cnt = np.bincount(ev["token"][m], minlength=V)
sm = np.bincount(ev["token"][m], weights=ll, minlength=V)
g_mean, g_std = ll.mean(), ll.std()
tl = np.where(cnt > 0, sm / np.maximum(cnt, 1), g_mean)
tl[0] = g_mean
tok_loglen = torch.tensor((tl - g_mean) / g_std, dtype=torch.float32)

mc = GPTRecConfig(vocab_size=V, max_len=L, n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"], n_embd=cfg["model"]["n_embd"])
model = RewardModel(mc, tok_loglen).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

names = ["g>=1", "g>=3", "g>=10", "next-day"]
v_idx, v_cs = eval_samples(ev, "val", 3, seed=123, L=L)
keep = ev["token"][v_idx] > 1
v_idx, v_cs = v_idx[keep], v_cs[keep]
vy, vok = ev["g"][v_idx], ev["g_ok"][v_idx]


def auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


# baselines: session age (known at the time of the event) and the user's own train-period next-day return rate
nu = len(ev["u_start"])
tr = m & ev["g_ok"][:, 3]
user_rate = np.bincount(ev["uidx"][tr], weights=ev["g"][tr, 3], minlength=nu) / np.maximum(np.bincount(ev["uidx"][tr], minlength=nu), 1)
print("val base rates:", {n: round(float(vy[vok[:, k], k].mean()), 3) for k, n in enumerate(names)})
print("baseline AUC (session age) for g>=1/3/10:", [round(auc(vy[:, k], ev["sess_pos"][v_idx]), 3) for k in range(3)],
      "| baseline AUC (user's own next-day rate):", round(auc(vy[vok[:, 3], 3], user_rate[ev["uidx"][v_idx]][vok[:, 3]]), 3))

ckpt_dir = ROOT / "checkpoints" / args.name
ckpt_dir.mkdir(parents=True, exist_ok=True)
log = open(ROOT / "reports/logs" / f"{args.name}.jsonl", "a")
starts, ends = train_windows(ev, rng, stride=L)
n_steps = (min(len(starts), args.max_windows) if args.max_windows else len(starts)) // args.bs
total, warm = n_steps * args.epochs, max(1, int(0.03 * n_steps * args.epochs))


def lr_at(s):
    if s < warm:
        return args.lr * (s + 1) / warm
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


best, step = -1.0, 0
for ep in range(args.epochs):
    if ep > 0:
        starts, ends = train_windows(ev, rng, stride=L)
    perm = rng.permutation(len(starts))
    model.train()
    t0, acc, k = time.time(), 0.0, 0
    for it in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sl = perm[it * args.bs:(it + 1) * args.bs]
        x, fb_in, y, ok = [t.to(dev) for t in make_long_batch(ev, starts[sl], ends[sl], L)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, fb_in)
        bce = F.binary_cross_entropy_with_logits(out.float(), y, reduction="none")
        loss = ((bce * ok).sum((0, 1)) / ok.sum((0, 1)).clamp(min=1)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        acc += loss.item()
        k += 1
        if step % 200 == 0:
            eta = (time.time() - t0) / (it + 1) * (n_steps - it - 1) / 60
            print(f"ep {ep} step {it+1}/{n_steps} loss {acc/k:.4f} | ETA {eta:.1f} min", flush=True)
            acc, k = 0.0, 0
    P = rm_predict(model, ev, v_idx, v_cs, L, dev)
    aucs = [auc(vy[vok[:, j], j], P[vok[:, j], j]) for j in range(4)]
    log.write(json.dumps({"epoch": ep, **dict(zip(names, aucs))}) + "\n"); log.flush()
    print(f"[val] epoch {ep}: AUC " + " | ".join(f"{n} {a:.4f}" for n, a in zip(names, aucs)))
    state = {"model": model.state_dict(), "cfg": asdict(mc), "epoch": ep}
    torch.save(state, ckpt_dir / "last.pt")
    if np.nanmean(aucs) > best:
        best = float(np.nanmean(aucs))
        torch.save(state, ckpt_dir / "best.pt")
print("done. best mean AUC:", best)