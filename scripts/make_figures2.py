import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ROOT

RES, FIG = ROOT / "reports/results", ROOT / "reports/figures"
panels = [("val_final_novelty_K1000", "награда зависит только от трека:\nlog10 ранга (без модели)"),
          ("val_final_itemmean", "награда зависит только от трека:\nсредняя доля прослушивания (train)"),
          ("val_final_big", "награда rm_a (трек + контекст)")]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (name, title) in zip(axs, panels):
    p = RES / f"{name}.json"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("средняя награда в топ-10"); ax.set_ylabel("Recall@10")
    if not p.exists():
        continue
    d = json.load(open(p, encoding="utf-8"))
    rr = sorted([k for k in d if k.startswith("rerank_a")], key=lambda k: float(k[8:]))
    ax.plot([d[k]["rA"][0] for k in rr], [d[k]["hit"][0] for k in rr], "o-", color="k", ms=3, label="переранжирование")
    for k in d:
        if k.startswith("rerank_a"):
            continue
        if name == "val_final_big" and not (k.startswith("RL_beta") or k.startswith("Distill_a")):
            continue
        y, x = d[k]["hit"], d[k]["rA"][0]
        ax.errorbar(x, y[0], yerr=[[y[0] - y[1]], [y[2] - y[0]]], fmt="s", capsize=3,
                    color="tab:red" if k.startswith("RL") else "tab:blue")
        ax.annotate(k.replace("RLnov_beta", "β=").replace("RLim_beta", "β=").replace("RL_beta", "β=").replace("Distill_a", "дист. α="),
                    (x, y[0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "gap_by_reward_type.png", dpi=130); plt.close()
print("written gap_by_reward_type.png")