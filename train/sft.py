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

from data.loader import eval_samples, load_events, make_batch, train_windows
from eval.evaluate import model_ranks
from eval.metrics import report
from models.gpt_rec import GPTRec, GPTRecConfig
from src.utils import ROOT, load_cfg, set_seed

cfg = load_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["all", "pos"], default="all")
ap.add_argument("--name", default=None)
ap.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
ap.add_argument("--bs", type=int, default=cfg["train"]["batch_size"])
ap.add_argument("--accum", type=int, default=cfg["train"]["grad_accum"])
ap.add_argument("--lr", type=float, default=cfg["train"]["lr"])
ap.add_argument("--seed", type=int, default=cfg["seed"])
ap.add_argument("--max_windows", type=int, default=0, help="debug: windows per epoch")
args = ap.parse_args()
name = args.name or f"base_{args.mode}"

set_seed(args.seed)
L, thr = cfg["data"]["max_len"], cfg["data"]["positive_ratio_threshold"]
V = cfg["data"]["vocab_size"] + 2
dev = "cuda"
rng = np.random.default_rng(args.seed)

ev = load_events()
mc = GPTRecConfig(vocab_size=V, max_len=L, **{k: cfg["model"][k] for k in ("n_layer", "n_head", "n_embd")})
model = GPTRec(mc).to(dev)
print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

v_idx, v_cs = eval_samples(ev, "val", 3, seed=123, L=L)       # small val sample for monitoring
ckpt_dir = ROOT / "checkpoints" / name
ckpt_dir.mkdir(parents=True, exist_ok=True)
log = open(ROOT / "reports/logs" / f"{name}.jsonl", "a")

starts, ends = train_windows(ev, rng, stride=L)
per_step = args.bs * args.accum
steps_per_epoch = (min(len(starts), args.max_windows) if args.max_windows else len(starts)) // per_step
total, warm = steps_per_epoch * args.epochs, max(1, int(0.03 * steps_per_epoch * args.epochs))
print(f"windows/epoch: {len(starts)}, steps/epoch: {steps_per_epoch}, mode: {args.mode}")


def lr_at(s):
    if s < warm:
        return args.lr * (s + 1) / warm
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


best, step = -1.0, 0
for ep in range(args.epochs):
    if ep > 0:
        starts, ends = train_windows(ev, rng, stride=L)       # new random offsets each epoch
    perm = rng.permutation(len(starts))[: steps_per_epoch * per_step]
    model.train()
    t0, run, n_tok = time.time(), 0.0, 0
    for it in range(steps_per_epoch):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for a in range(args.accum):
            sl = perm[(it * args.accum + a) * args.bs: (it * args.accum + a + 1) * args.bs]
            x, y = make_batch(ev, starts[sl], ends[sl], L, args.mode, thr)
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / args.accum).backward()
            run += loss.item() / args.accum
            n_tok += int((y != -100).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            el = time.time() - t0
            eta = el / (it + 1) * (steps_per_epoch - it - 1) / 60
            print(f"ep {ep} step {it+1}/{steps_per_epoch} loss {run/100:.4f} "
                  f"{n_tok/el:,.0f} target tok/s, epoch ETA {eta:.1f} min", flush=True)
            run = 0.0
    ranks, _ = model_ranks(model, ev, v_idx, v_cs, dev, L=L)
    rows, cov = report(ranks, ev, v_idx, thr)
    crit_group = "all" if args.mode == "all" else "positive"
    crit = rows[crit_group]["ndcg@10"]
    rec = {"epoch": ep, "val_all": rows["all"], "val_positive": rows["positive"]}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[val] epoch {ep}: all recall@10 {rows['all']['recall@10']:.4f} ndcg@10 {rows['all']['ndcg@10']:.4f} | "
          f"positive recall@10 {rows['positive']['recall@10']:.4f} ndcg@10 {rows['positive']['ndcg@10']:.4f}")
    state = {"model": model.state_dict(), "cfg": asdict(mc), "mode": args.mode, "epoch": ep}
    torch.save(state, ckpt_dir / "last.pt")
    if crit > best:
        best = crit
        torch.save(state, ckpt_dir / "best.pt")
print("done. best val ndcg@10:", best)