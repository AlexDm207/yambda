import numpy as np
import torch

from data.loader import make_eval_batch


@torch.no_grad()
def model_ranks(model, ev, tgt_idx, ctx_start, device="cuda", bs=256, L=128, topk=10):
    model.eval()
    n = len(tgt_idx)
    ranks = np.zeros(n, dtype=np.int64)
    tops = np.zeros((n, topk), dtype=np.int32)
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        x, lens, tgt = make_eval_batch(ev, tgt_idx[sl], ctx_start[sl], L)
        x = torch.from_numpy(x).long().to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden(x)
        last = torch.from_numpy(lens - 1).to(device)
        h = h[torch.arange(len(x), device=device), last]   # state after the last real event
        lg = h.float() @ model.tok.weight.float().T
        lg[:, :2] = float("-inf")
        t = torch.from_numpy(tgt).long().to(device)
        tl = lg.gather(1, t[:, None])
        ranks[sl] = ((lg > tl).sum(1) + 1).cpu().numpy()   # rank of the true item
        tops[sl] = lg.topk(topk, dim=1).indices.cpu().numpy()
    return ranks, tops


def popularity_ranks(tgt_tok):
    # tokens are assigned by train popularity, so rank = token - 1
    return np.maximum(tgt_tok - 1, 1)


def history_freq_ranks(ev, tgt_idx, ctx_start, L=128):
    """Rank items by how often they occur in the last L events (ties -> more popular first)."""
    x, lens, tgt = make_eval_batch(ev, tgt_idx, ctx_start, L)
    ranks = np.empty(len(tgt), dtype=np.int64)
    for i in range(len(tgt)):
        t = int(tgt[i])
        if t <= 1:
            ranks[i] = 10**9
            continue
        ctx = x[i, : lens[i]]
        uniq, cnt = np.unique(ctx[ctx > 1], return_counts=True)
        c_t = int(cnt[uniq == t].sum())
        if c_t > 0:
            higher = ((cnt > c_t) | ((cnt == c_t) & (uniq < t))).sum()
        else:  # all history items come first, then unseen items by popularity
            higher = len(uniq) + (t - 2) - (uniq < t).sum()
        ranks[i] = higher + 1
    return ranks


def rec_stats(tops, vocab_size):
    flat = tops.ravel()
    u, c = np.unique(flat, return_counts=True)
    return {
        "catalog_coverage@10": len(u) / (vocab_size - 2),
        "share_of_top100_popular": float((flat <= 101).mean()),
        "most_recommended_item_share": float(c.max() / len(flat)),
    }