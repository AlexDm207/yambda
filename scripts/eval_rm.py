import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.loader import eval_samples, load_events
from eval.rm_eval import rm_metrics, rm_predict
from models.gpt_rec import GPTRecConfig
from models.reward_model import RewardModel
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--ckpt", default="checkpoints/rm_a/best.pt")
ap.add_argument("--n_per_user", type=int, default=10)
args = ap.parse_args()

cfg = load_cfg()
L, V = cfg["data"]["max_len"], cfg["data"]["vocab_size"] + 2
ev = load_events()
idx, cs = eval_samples(ev, args.split, args.n_per_user, seed=2024, L=L)
keep = ev["token"][idx] > 1                      # only candidates a policy can recommend
idx, cs = idx[keep], cs[keep]
tok = ev["token"]
y = (ev["ratio"][idx] / 100.0, ev["like"][idx], ev["dislike"][idx])
print(f"{args.split}: {len(idx)} eval samples (in-vocabulary targets)")

ck = torch.load(ROOT / args.ckpt, map_location="cpu")
model = RewardModel(GPTRecConfig(**ck["cfg"]), torch.zeros(V)).to("cuda")
model.load_state_dict(ck["model"])
P = rm_predict(model, ev, idx, cs, L, "cuda")

# baselines: statistics from TRAIN only (+ the user's own context window)
tr = ev["period"] == 0
r = ev["ratio"].astype(np.float64) / 100.0
g = r[tr].mean()
cnt = np.bincount(tok[tr], minlength=V)
sm = np.bincount(tok[tr], weights=r[tr], minlength=V)
item_mean = np.where(cnt >= 5, sm / np.maximum(cnt, 1), g)
csum = np.concatenate([[0.0], np.cumsum(r)])
start = np.maximum(cs, idx - L + 1)
user_mean = (csum[idx] - csum[start]) / (idx - start)
base = {
    "global_mean": np.full(len(idx), g),
    "item_mean": item_mean[tok[idx]],
    "user_mean": user_mean,
    "user+item": np.clip(item_mean[tok[idx]] + user_mean - g, 0, 1),
}

cols = ["n", "ratio_mae", "ratio_rmse", "ratio_spearman", "comp_auc", "comp_ap", "comp_ece"]
res = {"reward_model": rm_metrics(*y, P[:, 0], P[:, 1], P[:, 2], P[:, 3])}
for k, v in base.items():
    res[k] = rm_metrics(*y, v)
print("\n=== overall ===")
print(pd.DataFrame(res).T.reindex(columns=cols).round(4).to_string())
m = res["reward_model"]
print(f"\nlike: AUC {m['like_auc']:.3f} PR-AUC {m['like_ap']:.4f} (base rate {m['like_rate']:.4f}, mean pred {m['like_mean_pred']:.4f}) | "
      f"dislike AUC {m['dis_auc']:.3f} | completed: base rate {m['comp_base_rate']:.3f}, mean pred {m['comp_mean_pred']:.3f}")

# slices: popularity, track length, repeat/new
ln = ev["length"][idx]
slices = {
    "top1k": tok[idx] <= 1001, "top1k-10k": (tok[idx] > 1001) & (tok[idx] <= 10001), "tail(10k+)": tok[idx] > 10001,
    "len<3min": ln < 180, "len3-4min": (ln >= 180) & (ln < 240), "len>4min": ln >= 240,
    "repeat": ev["is_repeat"][idx].astype(bool), "new": ~ev["is_repeat"][idx].astype(bool),
}
rows = {}
for name, mk in slices.items():
    a = rm_metrics(*[v[mk] for v in y], P[mk, 0], P[mk, 1], P[mk, 2], P[mk, 3])
    b = rm_metrics(*[v[mk] for v in y], base["user+item"][mk])
    rows[name] = {"n": a["n"], "base_rate": a["comp_base_rate"], "mae_rm": a["ratio_mae"], "mae_u+i": b["ratio_mae"],
                  "auc_rm": a["comp_auc"], "auc_u+i": b["comp_auc"]}
print("\n=== slices ===")
print(pd.DataFrame(rows).T.round(4).to_string())

out = ROOT / "reports/results"
json.dump({"overall": res, "slices": rows}, open(out / f"{args.split}_{Path(args.ckpt).parent.name}_rm.json", "w"), indent=2)