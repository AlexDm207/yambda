"""Markdown tables for the report from reports/results/*_final_*.json -> reports/tables.md"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

RES = ROOT / "reports/results"


def f(v, d=3):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


lines = []
for split, name in [("val", "val_final_main"), ("test", "test_final_test")]:
    p = RES / f"{name}.json"
    if not p.exists():
        continue
    d = json.load(open(p, encoding="utf-8"))
    lines += [f"\n### {split}\n",
              "| метод | seeds | R@10 | NDCG@10 | награда rm_a | награда rm_b | длина, с | топ-100 | макс. доля трека | покрытие | разрыв (rm_a) | разрыв (rm_b) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in d.items():
        ga = f(m["gap_rA"]) if "gap_rA" in m else ""
        gb = f(m["gap_rB"]) if "gap_rB" in m else ""
        ns = len(m.get("seeds_R10", [])) or 1
        lines.append(f"| {k} | {ns} | {f(m['hit'])} | {f(m['ndcg'])} | {m['rA'][0]:.4f} | {m['rB'][0]:.4f} | "
                     f"{m['len'][0]:.0f} | {m['top100'][0]:.3f} | {m['maxshare'][0]:.4f} | {m['coverage'][0]:.3f} | {ga} | {gb} |")
(ROOT / "reports/tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written reports/tables.md")