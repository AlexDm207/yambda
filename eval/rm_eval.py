import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from data.loader import make_rm_batch


@torch.no_grad()
def rm_predict(model, ev, tgt_idx, ctx_start, L=128, device="cuda", bs=512):
    """Probabilities [ratio, completed, like, dislike] for the event at tgt_idx given the previous <=L-1 events."""
    model.eval()
    start = np.maximum(ctx_start, tgt_idx - L + 1)
    P = np.zeros((len(tgt_idx), 4), dtype=np.float32)
    for i in range(0, len(tgt_idx), bs):
        sl = slice(i, i + bs)
        x, fb_in, *_ = make_rm_batch(ev, start[sl], tgt_idx[sl] + 1, L)
        last = torch.from_numpy(tgt_idx[sl] - start[sl]).to(device)       # last real position
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.to(device), fb_in.to(device))
        out = out[torch.arange(len(x), device=device), last]
        P[sl] = torch.sigmoid(out.float()).cpu().numpy()
    return P


def _safe(fn, y, s):
    try:
        return float(fn(y, s))
    except ValueError:
        return float("nan")


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def rm_metrics(y_ratio, y_like, y_dis, p_ratio, p_comp=None, p_like=None, p_dis=None):
    y_comp = (y_ratio >= 0.5).astype(int)
    out = {
        "n": int(len(y_ratio)),
        "ratio_mae": float(np.abs(p_ratio - y_ratio).mean()),
        "ratio_rmse": float(np.sqrt(((p_ratio - y_ratio) ** 2).mean())),
        "ratio_spearman": _safe(lambda a, b: spearmanr(a, b)[0], y_ratio, p_ratio),
        "comp_base_rate": float(y_comp.mean()),
    }
    s = p_comp if p_comp is not None else p_ratio     # baselines are scored by their ratio estimate
    out["comp_auc"] = _safe(roc_auc_score, y_comp, s)
    out["comp_ap"] = _safe(average_precision_score, y_comp, s)
    if p_comp is not None:
        out["comp_ece"] = _ece(p_comp, y_comp)
        out["comp_mean_pred"] = float(p_comp.mean())
    if p_like is not None:
        out["like_auc"] = _safe(roc_auc_score, y_like, p_like)
        out["like_ap"] = _safe(average_precision_score, y_like, p_like)
        out["like_rate"], out["like_mean_pred"] = float(y_like.mean()), float(p_like.mean())
    if p_dis is not None:
        out["dis_auc"] = _safe(roc_auc_score, y_dis, p_dis)
    return out