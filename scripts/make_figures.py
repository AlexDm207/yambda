"""Report figures. Input: reports/results/*.json, reports/logs/*_rl.jsonl. Output: reports/figures/*.png
Usage: python scripts/make_figures.py [val_final_main]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    VARIANT = ("RL_debiased", "RL_seconds", "RL_sft", "Distill_a30_long")       # single-seed variants, val only
    main = [k for k in others if not k.startswith(VARIANT)]
    variants = [k for k in others if k.startswith(VARIANT)]
    ref = [k for k in main if k in ("RL_beta0.2", "RL_beta0.1", "RL_beta0.05", "Distill_a30")]

    def panel(ax, xkey, ykey, ylabel, ci, keys, faded=(), final=final):
        fx, fy = [final[k][xkey][0] for k in rr], [final[k][ykey][0] for k in rr]
        ax.plot(fx, fy, "o-", color="k", ms=3, label="переранжирование топ-100 базы")
        obst = [(x0 + (x1 - x0) * t / 10, y0 + (y1 - y0) * t / 10)          # points along the frontier curve
                for (x0, y0), (x1, y1) in zip(zip(fx, fy), zip(fx[1:], fy[1:])) for t in range(11)]
        labels = [(final[k][xkey][0], final[k][ykey][0], "α=" + k[8:]) for k in rr if float(k[8:]) in (10, 30, 100)]
        for k in [*faded, *keys]:
            x, y = final[k][xkey][0], final[k][ykey]
            single = len(final[k].get("seeds_R10", [])) < 2                  # hollow marker: one seed only
            ax.errorbar(x, y[0], yerr=[[y[0] - y[1]], [y[2] - y[0]]] if ci else None, fmt="s", color=color(k),
                        capsize=3, mfc="white" if single else color(k), alpha=0.35 if k in faded else 1.0)
            obst += [(x, y[0] + (y[2] - y[1]) * (t / 4 - 0.5)) for t in range(5)] if ci else [(x, y[0])]
            labels.append((x, y[0], label(k)))
        ax.set_ylabel(ylabel)
        ax._labels, ax._obst = labels, obst

    def seed_legend(ax):
        h = [Line2D([], [], color="k", marker="o", ms=3, label="переранжирование топ-100 базы"),
             Line2D([], [], ls="", marker="s", color="0.3", label="среднее по 3 seed, 95% ДИ"),
             Line2D([], [], ls="", marker="s", color="0.3", mfc="white", label="1 seed, 95% ДИ")]
        ax.legend(handles=h, fontsize=7, loc="lower left")

    CAND = [(dx, dy) for dy in (0, 7, -7, 14, -14, 21, -21, 28, -28, 35, -35) for dx in (5, -5)]

    def place_labels(fig):
        """Greedy label placement: per label, the first candidate offset with the fewest collisions
        (other labels, markers, error bars, the frontier curve, the axes border)."""
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax in fig.axes:
            if not hasattr(ax, "_labels"):
                continue
            pts = ax.transData.transform(ax._obst + [(x, y) for x, y, _ in ax._labels])
            placed = []
            for x, y, text in ax._labels:
                best = None
                for dx, dy in CAND:
                    a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                                    ha="left" if dx > 0 else "right", va="center")
                    bb = a.get_window_extent(rend).expanded(1.04, 1.15)
                    bm = bb.padded(5)                                    # markers are ~10 px wide
                    cost = (10 * sum(bb.overlaps(p) for p in placed) + sum(bm.contains(px, py) for px, py in pts)
                            + 20 * (not (ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1))))
                    if best is None or cost < best[0]:
                        if best:
                            best[1].remove()
                        best = (cost, a, bb)
                    else:
                        a.remove()
                    if cost == 0:
                        break
                assert best is not None
                placed.append(best[2])
                dx, dy = best[1].xyann
                if abs(dy) >= 14:                                        # far from its point: thin leader line
                    best[1].remove()
                    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                                ha="left" if dx > 0 else "right", va="center",
                                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5", shrinkA=0, shrinkB=4))

    for name, keys, faded in [("pareto", main, ()), ("pareto_variants", variants, ref)]:
        if not keys:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, xk, t in zip(axs, ["rA", "rB"], ["награда по rm_a (на ней обучали)", "награда по rm_b (проверочная)"]):
            panel(ax, xk, "hit", "Recall@10", True, keys, faded)
            ax.set_xlabel("средняя предсказанная доля прослушивания в топ-10")
            ax.set_title(t)
        seed_legend(axs[0])
        if faded:
            fig.suptitle("Варианты награды и регуляризации (1 seed); бледные точки: основные методы", fontsize=10)
        plt.tight_layout(); place_labels(fig); plt.savefig(FIG / f"{stem}_{name}.png"); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, yk, yl in zip(axs, ["top100", "maxshare", "coverage"],
                          ["доля слотов 100 самых популярных треков", "доля слотов у самого частого трека", "покрытие каталога"]):
        panel(ax, "rA", yk, yl, False, others)
        ax.set_xlabel("награда по rm_a")
    plt.tight_layout(); place_labels(fig); plt.savefig(FIG / f"{stem}_behaviour.png"); plt.close()


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