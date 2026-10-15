"""One-step KL-regularised policy improvement by distillation.
Target per context: q(a|s) ~ pi_ref(a|s) * exp(alpha * r(s,a)) on the top-K candidates of pi_ref
(the closed-form optimum of E[r] - (1/alpha) KL(pi || pi_ref) restricted to the candidate set).
The policy is trained with cross-entropy to q."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.loader import load_events
from src.rl_lib import AC, REWARDS, combine, context_arrays, load_policy, load_rm, rm_score_candidates, token_stats
from src.utils import ROOT, load_cfg, set_seed

cfg = load_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--init", default="checkpoints/base_all/best.pt")
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--reward", choices=REWARDS, default="ratio")
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--target", choices=["cand", "full"], default="cand")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--log_every", type=int, default=20)
ap.add_argument("--save_every", type=int, default=200)
ap.add_argument("--seed", type=int, default=cfg["seed"])
args = ap.parse_args()

set_seed(args.seed)
dev = "cuda"
rng = np.random.default_rng(args.seed)
L, V = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
C = L - 1

ev = load_events()
policy, pcfg = load_policy(args.init)
ref, _ = load_policy(args.init)
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval()
ref.eval()
rm = load_rm(args.rm_a)
len_sec, len_bias = token_stats(ev, V, args.reward)
Wr = ref.tok.weight[2:].float().T.contiguous()
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98))

a, b = ev["bounds"]["train"]
lo = np.maximum(a, ev["u_start"] + C)
cnt = np.maximum(b - lo, 0)
prob = cnt / cnt.sum()


def sample(n):
    u = rng.choice(len(cnt), size=n, p=prob)
    return lo[u] + (rng.random(n) * cnt[u]).astype(np.int64)


ckpt_dir = ROOT / "checkpoints" / args.name
ckpt_dir.mkdir(parents=True, exist_ok=True)
log = open(ROOT / "reports/logs" / f"{args.name}_distill.jsonl", "a")
print(f"distill: reward={args.reward} alpha={args.alpha} K={args.K} B={args.B} lr={args.lr}")


def save(name, step):
    torch.save({"model": policy.state_dict(), "cfg": pcfg, "mode": "distill", "step": step, "args": vars(args)},
               ckpt_dir / name)


acc, n_acc, t0 = {}, 0, time.time()
for step in range(1, args.steps + 1):
    idx = sample(args.B)
    tok, fb_in, fb_last = context_arrays(ev, idx, C)
    x = torch.from_numpy(tok).long().to(dev)

    with torch.no_grad():
        with torch.autocast(**AC):
            h = ref.hidden(x)[:, -1]
        lpr = torch.log_softmax(h.float() @ Wr, dim=-1)
        lpr_k, ik = lpr.topk(args.K, dim=-1)                     # reference top-K candidates
    cands = (ik + 2).cpu().numpy()
    P = rm_score_candidates(rm, tok, fb_in, fb_last, cands, dev)  # (B, K, 4)
    r = torch.from_numpy(combine(P, cands, args.reward, len_sec, len_bias)).float().to(dev)
    q = torch.softmax(lpr_k + args.alpha * r, dim=-1)             # target distribution on candidates

    with torch.autocast(**AC):
        h = policy.hidden(x)[:, -1]
    lp = torch.log_softmax(h.float() @ policy.tok.weight[2:].float().T, dim=-1)
    lp_k = lp.gather(1, ik)
    if args.target == "full":
        q_full = lpr.exp().clone()
        q_full.scatter_(1, ik, q * lpr_k.exp().sum(1, keepdim=True))   # keep the reference mass of the candidate set
        loss = -(q_full * lp).sum(1).mean()
    else:
        loss = -(q * lp_k).sum(1).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    with torch.no_grad():
        st = {
            "ce": loss.item(),
            "H_target": -(q * torch.log(q + 1e-12)).sum(1).mean().item(),
            "r_ref": (torch.softmax(lpr_k, -1) * r).sum(1).mean().item(),
            "r_target": (q * r).sum(1).mean().item(),
            "r_policy": (torch.softmax(lp_k, -1) * r).sum(1).mean().item(),
            "mass_topK": lp_k.exp().sum(1).mean().item(),
        }
    for k, v in st.items():
        acc[k] = acc.get(k, 0.0) + v
    n_acc += 1
    if step == 1 or step % args.log_every == 0:
        m = {k: v / n_acc for k, v in acc.items()}
        gain = (m["r_policy"] - m["r_ref"]) / max(m["r_target"] - m["r_ref"], 1e-9)
        print(f"step {step:4d} | CE {m['ce']:.3f} H(q) {m['H_target']:.3f} | r_ref {m['r_ref']:.4f} "
              f"r_target {m['r_target']:.4f} r_policy {m['r_policy']:.4f} (achieved {gain:.0%}) | "
              f"mass_top{args.K} {m['mass_topK']:.3f} | {(time.time() - t0) / step:.2f}s/step", flush=True)
        log.write(json.dumps({"step": step, **m, "gain": gain}) + "\n"); log.flush()
        acc, n_acc = {}, 0
    if step % args.save_every == 0:
        save(f"step{step}.pt", step)
save("last.pt", args.steps)
print("done")