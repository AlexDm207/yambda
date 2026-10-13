import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from data.loader import eval_samples, load_events
from eval.evaluate import history_freq_ranks, model_ranks, popularity_ranks, rec_stats
from eval.metrics import report
from models.gpt_rec import GPTRec, GPTRecConfig
from src.utils import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--ckpt", default=None, help="e.g. checkpoints/base_all/best.pt; empty = baselines")
ap.add_argument("--n_per_user", type=int, default=10)
args = ap.parse_args()

cfg = load_cfg()
thr, L = cfg["data"]["positive_ratio_threshold"], cfg["data"]["max_len"]
ev = load_events()
idx, cs = eval_samples(ev, args.split, args.n_per_user, seed=2024, L=L)   # same samples for everyone
print(f"{args.split}: {len(idx)} eval samples")

cols = ["n", "recall@10", "ndcg@10", "map@10", "recall@50", "ndcg@50"]
out_dir = ROOT / "reports/results"


def show(name, ranks, tops=None):
    rows, cov = report(ranks, ev, idx, thr)
    print(f"\n=== {name} | {args.split} | targets inside vocabulary: {cov:.1%} ===")
    print(pd.DataFrame(rows).T[cols].round(4).to_string())
    res = {"groups": rows, "vocab_coverage": cov}
    if tops is not None:
        res["rec_stats"] = rec_stats(tops, cfg["data"]["vocab_size"] + 2)
        print(res["rec_stats"])
    json.dump(res, open(out_dir / f"{args.split}_{name}.json", "w"), indent=2)


if args.ckpt:
    ck = torch.load(ROOT / args.ckpt, map_location="cpu")
    model = GPTRec(GPTRecConfig(**ck["cfg"])).to("cuda")
    model.load_state_dict(ck["model"])
    ranks, tops = model_ranks(model, ev, idx, cs, "cuda", L=L)
    show(Path(args.ckpt).parent.name, ranks, tops)
else:
    show("popularity", popularity_ranks(ev["token"][idx]))
    show("history_freq", history_freq_ranks(ev, idx, cs, L))