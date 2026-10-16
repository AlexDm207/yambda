"""Report figures. Input: reports/results/*.json, reports/logs/*_rl.jsonl. Output: reports/figures/*.png
Usage: python scripts/make_figures.py [val_final_main]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ROOT

RES, LOG, FIG = ROOT / "reports/results", ROOT / "reports/logs", ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
stem = sys.argv[1] if len(sys.argv) > 1 else "val_final_main"


def load(path):
    if not Path(path).exists():
        print("missing:", Path(path).name)
        return None
    return json.load(open(path, encoding="utf-8"))


def color(k):
    return "tab:red" if k.startswith("RL") else "tab:blue" if k.startswith("Distill") else "tab:green"


def label(k):
    return k.replace("RL_beta", "RL β=").replace("Distill_a", "дист. α=")


# ---- 1-2. relevance/reward frontier and behaviour ----
final = load(RES / f"{stem}.json")
if final:
    rr = sorted([k for k in final if k.startswith("rerank_a")], key=lambda k: float(k[8:]))
    others = [k for k in final if not k.startswith("rerank_")]

    offsets = {                                                        # label offsets (points) for the Pareto plot
        "Distill_a10": (5, 6), "RL_sft1_b0.2": (-6, 1), "RL_beta0.4": (-6, -16), "Distill_a30": (7, 3),
        "Distill_a30_long": (7, -11), "Base_pos": (-6, -12), "RL_debiased_b0.1": (-7, 3),
        "RL_sft0_b0.2": (-7, -6), "RL_beta0.1": (-7, -14), "RL_debiased_b0.05": (-7, -4), "RL_beta0.05": (-7, -4)}

    def panel(ax, xkey, ykey, ylabel, ci, final=final):
        ax.plot([final[k][xkey][0] for k in rr], [final[k][ykey][0] for k in rr], "o-", color="k", ms=3,
                label="переранжирование топ-100 базы")
        for k in rr:
            if float(k[8:]) in (10, 30, 100):
                ax.annotate("α=" + k[8:], (final[k][xkey][0], final[k][ykey][0]), fontsize=7,
                            xytext=(3, 4), textcoords="offset points")
        for k in others:
            x, y = final[k][xkey][0], final[k][ykey]
            ax.errorbar(x, y[0], yerr=[[y[0] - y[1]], [y[2] - y[0]]] if ci else None, fmt="s", color=color(k), capsize=3)
            ax.annotate(label(k), (x, y[0]), fontsize=7, xytext=offsets.get(k, (4, -9)) if ci else (4, -9),
                        textcoords="offset points", ha="right" if ci and offsets.get(k, (0,))[0] < 0 else "left")
        ax.set_ylabel(ylabel)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, xk, t in zip(axs, ["rA", "rB"], ["награда по rm_a (на ней обучали)", "награда по rm_b (проверочная)"]):
        panel(ax, xk, "hit", "Recall@10", True)
        ax.set_xlabel("средняя предсказанная доля прослушивания в топ-10")
        ax.set_title(t)
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / f"{stem}_pareto.png"); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, yk, yl in zip(axs, ["top100", "maxshare", "coverage"],
                          ["доля слотов 100 самых популярных треков", "доля слотов у самого частого трека", "покрытие каталога"]):
        panel(ax, "rA", yk, yl, False)
        ax.set_xlabel("награда по rm_a")
    plt.tight_layout(); plt.savefig(FIG / f"{stem}_behaviour.png"); plt.close()


# ---- 3. collapse trajectories (training logs, seed 42) ----
def read_log(name):
    p = LOG / f"{name}_rl.jsonl"
    if not p.exists():
        print("missing:", p.name)
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    first = [i for i, r in enumerate(rows) if r["step"] == 1]
    return rows[first[-1]:] if first else rows


runs = {"β=0.4": "rlstd_beta0.4", "β=0.2": "rlstd_beta0.2", "β=0.1": "rlstd_beta0.1", "β=0.05": "rl_ratio_b05"}
fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
for lab, name in runs.items():
    rows = read_log(name)
    if not rows:
        continue
    for ax, key, yl in zip(axs.ravel(), ["reward_A", "kl", "entropy", "uniq", "len_sec", "logrank"],
                           ["награда rm_a (среднее за 10 шагов)", "KL к базовой модели", "энтропия политики",
                            "доля уникальных выбранных треков", "длина выбранных треков, с", "log10(ранг популярности)"]):
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=lab)
        ax.set_ylabel(yl); ax.set_xlabel("шаг")
axs[0, 0].legend()
fig.suptitle("Траектории обучения RL, seed 42 (моменты скачков при β=0.1 зависят от seed)", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "collapse_trajectory.png"); plt.close()

# ---- 4. multi-step drift ----
ms = {"base_all": "base_all_best", "RL 1 шаг β=0.2": "rlstd_long_beta0.2_last",
      "RL 5 шагов β=0.4": "rlms_beta0.4_last", "RL 5 шагов β=0.2": "rlms_beta0.2_last"}
data = {l: load(RES / f"val_multistep_{t}_norepeat.json") for l, t in ms.items()}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
base = data["base_all"]
if base:
    st = [r["step"] for r in base]
    for key, lab in [("R10_teacher", "реальные треки в контексте"), ("R10_ctxonly", "только исходный контекст"),
                     ("R10_free", "собственные рекомендации в контексте")]:
        axs[0].plot(st, [r[key] for r in base], "o-", label=lab)
    axs[0].set_title("base_all: Recall@10 на шаге j"); axs[0].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.9))
    for key in ["rm_real", "rm_sim", "rm_zero", "rm_pos"]:
        axs[2].plot(st, [r[key] for r in base], "o-", label=key)
    axs[2].set_title("base_all: награда при разных допущениях", fontsize=9); axs[2].legend(fontsize=7)
for lab, rows in data.items():
    if rows:
        axs[1].plot([r["step"] for r in rows], [r["R10_free"] for r in rows], "o-", label=lab)
axs[1].set_title("свободная генерация, все политики"); axs[1].legend(fontsize=7)
for ax in axs:
    ax.set_xlabel("шаг генерации j")
plt.tight_layout(); plt.savefig(FIG / "multistep_drift.png"); plt.close()

# ---- 5. reward variants via reranking ----
files = {"ratio": "val_reward_base_all_best_ratio_rerank_long.json",
         "comp": "val_reward_base_all_best_comp_rerank_comp.json",
         "debiased": "val_reward_base_all_best_debiased_rerank_debiased.json",
         "composite": "val_reward_base_all_best_composite_rerank_composite.json",
         "seconds": "val_reward_base_all_best_seconds_rerank_seconds.json",
         "itemdebiased": "val_reward_base_all_best_itemdebiased_rerank_itemdebiased.json"}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for name, f in files.items():
    rows = load(RES / f)
    if not rows:
        continue
    for ax, (xk, yk) in zip(axs, [("pA_ratio", "R@10"), ("pA_ratio", "len_sec"), ("R@10", "pA_like")]):
        ax.plot([r[xk] for r in rows], [r[yk] for r in rows], "o-", ms=3, label=name)
axs[0].set(xlabel="предсказанная доля прослушивания в топ-10", ylabel="Recall@10")
axs[1].set(xlabel="предсказанная доля прослушивания в топ-10", ylabel="средняя длина треков, с")
axs[2].set(xlabel="Recall@10", ylabel="предсказанная вероятность лайка")
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "reward_variants.png"); plt.close()

# ---- 6. long-term reward ----
rows = load(RES / "val_long_rerank.json")
if rows:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key in zip(axs, ["g>=3", "g>=10", "next_day"]):
        for mode, lab in [("ratio", "по немедленной награде"), ("long", "по долгосрочной модели")]:
            rr_ = [r for r in rows if r["mode"] == mode]
            ax.plot([r["R@10"] for r in rr_], [r[key] for r in rr_], "o-", label=lab)
        y0 = [r for r in rows if r["mode"] == "long"][0][key]
        top = max(r[key] for r in rows)
        ax.set_ylim(y0 - 0.05, y0 + 0.05)                              # common scale: +-0.05 around the baseline
        ax.set_title(f"максимальный сдвиг: +{top - y0:.3f}", fontsize=9)
        ax.set_xlabel("Recall@10"); ax.set_ylabel("предсказанная вероятность: " + key)
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "long_term.png"); plt.close()
print("figures written to", FIG)