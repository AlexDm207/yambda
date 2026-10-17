"""Markdown tables for the report -> reports/tables.md"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

RES = ROOT / "reports/results"
SECTIONS = [
    ("val, 20000 контекстов (основная)", "val_final_big", "rm_a: предсказанная доля прослушивания", True),
    ("test, 20000 контекстов (основная)", "test_final_big", "rm_a: предсказанная доля прослушивания", True),
    ("test, 6000 контекстов (проверка устойчивости)", "test_final_test", "rm_a: предсказанная доля прослушивания", True),
    ("val: награда без модели, log10 ранга популярности (K=1000)", "val_final_novelty_K1000", "log10(ранг популярности)", False),
    ("val: награда без модели, средняя доля прослушивания трека в train (K=1000)", "val_final_itemmean", "средняя доля прослушивания трека", False),
    ("val: награда itemdebiased (RL)", "val_final_itemdeb", "rm_a: предсказанная доля прослушивания", False),
    ("val: бюджет выборки в RL", "val_final_batch", "rm_a: предсказанная доля прослушивания", False),
]


def fmt(v, d=3):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


lines = []
for title, name, axis, two in SECTIONS:
    p = RES / f"{name}.json"
    if not p.exists():
        lines.append(f"\n### {title}\n\n(нет файла {name}.json)\n")
        continue
    d = json.load(open(p, encoding="utf-8"))
    head = "| метод | seeds | R@10 | NDCG@10 | реальная доля прослушивания на попаданиях | награда | длина, с | топ-100 | макс. доля трека | покрытие | разрыв (ось награды) |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    if two:
        head += " разрыв (rm_b) |"
        sep += "---|"
    lines += [f"\n### {title}\n", f"Ось награды: {axis}.\n", head, sep]
    for k, m in d.items():
        rh = fmt(m["rph"]) if "rph" in m else ""
        ga = fmt(m["gap_rA"]) if "gap_rA" in m else ""
        row = (f"| {k} | {len(m.get('seeds_R10', [])) or 1} | {fmt(m['hit'])} | {fmt(m['ndcg'])} | {rh} | "
               f"{m['rA'][0]:.4f} | {m['len'][0]:.0f} | {m['top100'][0]:.3f} | {m['maxshare'][0]:.4f} | {m['coverage'][0]:.3f} | {ga} |")
        if two:
            row += (fmt(m["gap_rB"]) if "gap_rB" in m else "") + " |"
        lines.append(row)
(ROOT / "reports/tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written reports/tables.md")