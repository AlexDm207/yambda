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
from sklearn.metrics import roc_auc_score

from data.loader import eval_samples, load_events, make_rm_batch, train_windows
from eval.rm_eval import rm_metrics, rm_predict
from models.gpt_rec import GPTRecConfig
from models.reward_model import RewardModel, rm_loss
from src.utils import ROOT, load_cfg, set_seed

cfg = load_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="rm_a")
ap.add_argument("--seed", type=int, default=cfg["seed"])
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--n_layer", type=int, default=cfg["model"]["n_layer"])
ap.add_argument("--n_head", type=int, default=cfg["model"]["n_head"])
ap.add_argument("--n_embd", type=int, default=cfg["model"]["n_embd"])
ap.add_argument("--max_windows", type=int, default=0, help="debug")
args = ap.parse_args()

set_seed(args.seed)
L, V, dev = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2, "cuda"
rng = np.random.default_rng(args.seed)
ev = load_events()

# standardized log track length per token, from TRAIN events only
m = ev["period"] == 0
ll = np.log1p(ev["length"][m].astype(np.float64))
cnt = np.bincount(ev["token"][m], minlength=V)
sm = np.bincount(ev["token"][m], weights=ll, minlength=V)
g_mean, g_std = ll.mean(), ll.std()
tl = np.where(cnt > 0, sm / np.maximum(cnt, 1), g_mean)
tl[0] = g_mean
tok_loglen = torch.tensor((tl - g_mean) / g_std, dtype=torch.float32)

mc = GPTRecConfig(vocab_size=V, max_len=L, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
model = RewardModel(mc, tok_loglen).to(dev)
print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

v_idx, v_cs = eval_samples(ev, "val", 3, seed=123, L=L)
keep = ev["token"][v_idx] > 1
v_idx, v_cs = v_idx[keep], v_cs[keep]
vy = (ev["ratio"][v_idx] / 100.0, ev["like"][v_idx], ev["dislike"][v_idx])

ckpt_dir = ROOT / "checkpoints" / args.name
ckpt_dir.mkdir(parents=True, exist_ok=True)
log = open(ROOT / "reports/logs" / f"{args.name}.jsonl", "a")

starts, ends = train_windows(ev, rng, stride=L)
n_steps = (min(len(starts), args.max_windows) if args.max_windows else len(starts)) // args.bs
total, warm = n_steps * args.epochs, max(1, int(0.03 * n_steps * args.epochs))
print(f"windows/epoch: {len(starts)}, steps/epoch: {n_steps}")


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
    t0, acc, k = time.time(), np.zeros(4), 0
    for it in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        sl = perm[it * args.bs:(it + 1) * args.bs]
        x, fb_in, yr, yl, yd, mask = [t.to(dev) for t in make_rm_batch(ev, starts[sl], ends[sl], L)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, fb_in)
        loss, parts = rm_loss(out.float(), yr, yl, yd, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        acc += parts
        k += 1
        if step % 200 == 0:
            a = acc / k
            eta = (time.time() - t0) / (it + 1) * (n_steps - it - 1) / 60
            print(f"ep {ep} step {it+1}/{n_steps} ratio {a[0]:.4f} comp {a[1]:.4f} like {a[2]:.4f} dis {a[3]:.4f} | ETA {eta:.1f} min", flush=True)
            acc, k = np.zeros(4), 0
    P = rm_predict(model, ev, v_idx, v_cs, L, dev)
    r = rm_metrics(*vy, P[:, 0], P[:, 1], P[:, 2], P[:, 3])
    log.write(json.dumps({"epoch": ep, **r}) + "\n"); log.flush()
    print(f"[val] epoch {ep}: comp_auc {r['comp_auc']:.4f} ratio_mae {r['ratio_mae']:.4f} "
          f"ratio_spearman {r['ratio_spearman']:.4f} comp_ece {r['comp_ece']:.4f} like_ap {r['like_ap']:.4f}")
    state = {"model": model.state_dict(), "cfg": asdict(mc), "epoch": ep}
    torch.save(state, ckpt_dir / "last.pt")
    if r["comp_auc"] > best:
        best = r["comp_auc"]
        torch.save(state, ckpt_dir / "best.pt")
print("done. best val comp_auc:", best)