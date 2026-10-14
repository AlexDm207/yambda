"""On-policy REINFORCE with leave-one-out group baseline, exact KL to the reference policy and an SFT anchor."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.loader import load_events
from src.rl_lib import AC, REWARDS, combine, context_arrays, load_policy, load_rm, rm_rollout, rollout, token_stats
from src.utils import ROOT, load_cfg, set_seed

cfg = load_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--init", default="checkpoints/base_all/best.pt")
ap.add_argument("--rm_a", default="checkpoints/rm_a/best.pt")
ap.add_argument("--rm_b", default="checkpoints/rm_b/best.pt")
ap.add_argument("--reward", choices=REWARDS, default="ratio")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--B", type=int, default=32, help="contexts per step")
ap.add_argument("--G", type=int, default=16, help="samples per context")
ap.add_argument("--H", type=int, default=1, help="generated items per sample")
ap.add_argument("--gamma", type=float, default=0.9)
ap.add_argument("--beta", type=float, default=0.05, help="KL coefficient")
ap.add_argument("--sft", type=float, default=0.1, help="SFT anchor coefficient")
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--save_every", type=int, default=100)
ap.add_argument("--seed", type=int, default=cfg["seed"])
args = ap.parse_args()

set_seed(args.seed)
dev = "cuda"
rng = np.random.default_rng(args.seed)
L, V = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
B, G, H = args.B, args.G, args.H
C, M = L - H, args.B * args.G

ev = load_events()
policy, pcfg = load_policy(args.init)
ref, _ = load_policy(args.init)
for p in ref.parameters():
    p.requires_grad_(False)
policy.eval()   # no dropout: sampling and training use the same distribution
ref.eval()
rm_a = load_rm(args.rm_a)
rm_b = load_rm(args.rm_b) if args.rm_b else None
len_sec, len_bias = token_stats(ev, V)
len_sec_t = torch.tensor(len_sec, device=dev)
len_bias_t = torch.tensor(len_bias, device=dev)
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.98))

# training contexts: uniformly over TRAIN events with a full context before them
a, b = ev["bounds"]["train"]
lo = np.maximum(a, ev["u_start"] + C)
cnt = np.maximum(b - lo, 0)
prob = cnt / cnt.sum()


def sample(n):
    u = rng.choice(len(cnt), size=n, p=prob)
    return lo[u] + (rng.random(n) * cnt[u]).astype(np.int64)


ckpt_dir = ROOT / "checkpoints" / args.name
ckpt_dir.mkdir(parents=True, exist_ok=True)
log = open(ROOT / "reports/logs" / f"{args.name}_rl.jsonl", "a")
print(f"reward={args.reward} H={H} C={C} B={B} G={G} beta={args.beta} sft={args.sft} lr={args.lr}")


def save(name, step):
    torch.save({"model": policy.state_dict(), "cfg": pcfg, "mode": "rl", "step": step, "args": vars(args)},
               ckpt_dir / name)


acc, n_acc, t0 = {}, 0, time.time()
for step in range(1, args.steps + 1):
    idx = sample(B)
    tok, fb_in, fb_last = context_arrays(ev, idx, C)
    rep = lambda arr: torch.from_numpy(arr).to(dev).repeat_interleave(G, 0)
    ctx, fb_in_t, fb_last_t = rep(tok).long(), rep(fb_in), rep(fb_last)
    real_next = rep(ev["token"][idx]).long()

    # ---- rollout and rewards (no gradients) ----
    x = rollout(policy, ctx, H)                                   # (M, C+H)
    gen = x[:, C:]
    r = combine(rm_rollout(rm_a, x, fb_in_t, fb_last_t, C, H), gen, args.reward, len_sec_t, len_bias_t)
    rb = combine(rm_rollout(rm_b, x, fb_in_t, fb_last_t, C, H), gen, args.reward, len_sec_t, len_bias_t) if rm_b else r

    # ---- advantages: return-to-go minus leave-one-out group mean, scaled by batch std ----
    R, run = torch.zeros_like(r), torch.zeros(M, device=dev)
    for t in reversed(range(H)):
        run = r[:, t] + args.gamma * run
        R[:, t] = run
    Rg = R.view(B, G, H)
    A = ((Rg - Rg.mean(1, keepdim=True)) * G / (G - 1)).reshape(M, H)
    A = A / (A.std() + 1e-6)

    # ---- policy update (chunks; gradients accumulate) ----
    opt.zero_grad(set_to_none=True)
    kl_sum = ent_sum = nll_sum = 0.0
    for s in range(0, M, args.chunk):
        sl = slice(s, s + args.chunk)
        xs = x[sl]
        with torch.autocast(**AC):
            h = policy.hidden(xs[:, :-1])[:, C - 1:]              # positions that predict the generated items
        lp = torch.log_softmax(h.float() @ policy.tok.weight[2:].float().T, dim=-1)      # (m, H, V-2)
        logp_a = lp.gather(-1, (xs[:, C:] - 2).unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            with torch.autocast(**AC):
                hr = ref.hidden(xs[:, :-1])[:, C - 1:]
            lpr = torch.log_softmax(hr.float() @ ref.tok.weight[2:].float().T, dim=-1)
        pr = lp.exp()
        kl = (pr * (lp - lpr)).sum(-1)                            # exact KL(policy || reference) per position
        ent = -(pr * lp).sum(-1)
        rn = real_next[sl]
        nll = -lp[:, 0].gather(-1, (rn.clamp(min=2) - 2).unsqueeze(-1)).squeeze(-1) * (rn > 1)
        loss = (-(A[sl] * logp_a).sum() + args.beta * kl.sum()) / (M * H) + args.sft * nll.sum() / M
        loss.backward()
        kl_sum += kl.sum().item(); ent_sum += ent.sum().item(); nll_sum += nll.sum().item()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    st = {"reward_A": r.mean().item(), "reward_B": rb.mean().item(), "kl": kl_sum / (M * H),
          "entropy": ent_sum / (M * H), "sft_nll": nll_sum / M,
          "logrank": torch.log10(gen.float() - 1).mean().item(), "len_sec": len_sec_t[gen].mean().item(),
          "in_ctx": (gen[:, :, None] == ctx[:, None, :]).any(-1).float().mean().item(),
          "uniq": torch.unique(gen).numel() / gen.numel()}
    for k, v in st.items():
        acc[k] = acc.get(k, 0.0) + v
    n_acc += 1
    if step == 1 or step % args.log_every == 0:
        m = {k: v / n_acc for k, v in acc.items()}
        print(f"step {step:4d} | rA {m['reward_A']:.4f} rB {m['reward_B']:.4f} | KL {m['kl']:.4f} ent {m['entropy']:.2f} "
              f"sft {m['sft_nll']:.3f} | logrank {m['logrank']:.2f} len {m['len_sec']:.0f}s in_ctx {m['in_ctx']:.3f} "
              f"uniq {m['uniq']:.3f} | {(time.time() - t0) / step:.2f}s/step", flush=True)
        log.write(json.dumps({"step": step, **m}) + "\n"); log.flush()
        acc, n_acc = {}, 0
    if step % args.save_every == 0:
        save(f"step{step}.pt", step)
save("last.pt", args.steps)
print("done")