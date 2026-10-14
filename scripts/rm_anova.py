"""How context-dependent is the reward? Two-way ANOVA of RM predictions on a (contexts x items) grid."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.loader import eval_samples, load_events
from src.rl_lib import context_arrays, load_rm, rm_score_candidates
from src.utils import load_cfg

cfg = load_cfg()
L = cfg["data"]["max_len"]
C = L - 1
ev = load_events()
idx, cs = eval_samples(ev, "val", 10, seed=2024, L=L)
idx = idx[(idx - cs) >= C]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(idx, 600, replace=False))
tok, fb_in, fb_last = context_arrays(ev, idx, C)
items = rng.choice(np.arange(2, 10002), 200, replace=False)      # random items from the top-10k
cands = np.tile(items, (len(idx), 1))
P = rm_score_candidates(load_rm("checkpoints/rm_a/best.pt"), tok, fb_in, fb_last, cands)

for k, name in enumerate(["ratio", "completed", "like", "dislike"]):
    Y = P[..., k]
    g = Y.mean()
    r = Y.mean(1, keepdims=True) - g          # context (user) effect
    c = Y.mean(0, keepdims=True) - g          # item effect
    res = Y - g - r - c                       # interaction
    tot = ((Y - g) ** 2).sum()
    print(f"{name:10s} context {(r**2).sum() * Y.shape[1] / tot:6.1%} | item {(c**2).sum() * Y.shape[0] / tot:6.1%} "
          f"| interaction {(res**2).sum() / tot:6.1%}")