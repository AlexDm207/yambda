import numpy as np

KS = (1, 5, 10, 20, 50)


def summarize(ranks, ks=KS):
    out = {"n": int(len(ranks))}
    if len(ranks) == 0:
        return out
    r = ranks.astype(np.float64)
    for k in ks:
        hit = r <= k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(r + 1.0), 0.0).mean())
        out[f"map@{k}"] = float(np.where(hit, 1.0 / r, 0.0).mean())
    return out


def report(ranks, ev, tgt_idx, thr=50, ks=KS):
    inv = ev["token"][tgt_idx] > 1                    # target is inside the vocabulary
    pos = ev["ratio"][tgt_idx] >= thr
    rep = ev["is_repeat"][tgt_idx].astype(bool)
    groups = {"all": inv, "positive": inv & pos, "repeat": inv & rep, "new": inv & ~rep}
    return {g: summarize(ranks[m], ks) for g, m in groups.items()}, float(inv.mean())