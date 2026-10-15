"""events.parquet -> flat numpy arrays, training windows, evaluation samples."""
import numpy as np
import polars as pl
import torch

from src.utils import ROOT, load_cfg

# 0 train | 1 gap | 2 val | 3 gap | 4 test  (two separate gaps keep the codes monotonic in time)
PERIOD_CODE = {"train": 0, "gap1": 1, "val": 2, "gap2": 3, "test": 4}
N_PERIODS = len(PERIOD_CODE)


def _period_from_time(ts, S, day=86_400):
    # same boundaries as in prepare.py
    return np.select(
        [ts < S["train_end_day"] * day,
         ts < S["val_start_day"] * day,
         ts < S["val_end_day"] * day,
         ts < S["test_start_day"] * day],
        [0, 1, 2, 3], default=4,
    ).astype(np.uint8)


def _period_bounds(ev):
    # events are sorted by (uid, time) and the period code never decreases in time,
    # so each period is a contiguous range inside every user's block
    p, us, ue = ev["period"], ev["u_start"], ev["u_end"]
    cuts = []
    for k in range(N_PERIODS + 1):
        cs = np.concatenate([[0], np.cumsum(p < k)])
        cuts.append(us + (cs[ue] - cs[us]))
    return {name: (cuts[k], cuts[k + 1]) for name, k in PERIOD_CODE.items()}


def load_events():
    df = pl.read_parquet(
        ROOT / "data/processed/events.parquet",
        columns=["uid", "timestamp", "item_id", "ratio", "like", "dislike", "length", "token"],    ).with_columns(
        (~pl.col("item_id").is_first_distinct().over("uid")).alias("is_repeat"),
    )
    uid = df["uid"].to_numpy().astype(np.int64)
    change = np.flatnonzero(np.diff(uid)) + 1
    ev = {
        "token": df["token"].to_numpy().astype(np.int32),
        "ratio": df["ratio"].to_numpy(),
        "like": df["like"].to_numpy(),
        "is_repeat": df["is_repeat"].to_numpy(),
        "period": _period_from_time(df["timestamp"].to_numpy().astype(np.int64), load_cfg()["split"]),
        "u_start": np.concatenate([[0], change]),
        "u_end": np.concatenate([change, [len(uid)]]),
        "dislike": df["dislike"].to_numpy(),
        "length": df["length"].to_numpy(),
    }
    d = np.diff(ev["period"].astype(np.int8))
    d[ev["u_end"][:-1] - 1] = 0          # ignore the transition between two users
    assert (d >= 0).all(), "period must be non-decreasing inside a user"
    ev["bounds"] = _period_bounds(ev)
    return ev


def train_windows(ev, rng, stride=128):
    """Non-overlapping windows over each user's TRAIN events, random offset per user."""
    a, b = ev["bounds"]["train"]
    starts, ends = [], []
    for ua, ub in zip(a, b):
        if ub - ua < 2:
            continue
        off = int(rng.integers(0, stride))
        first = ua + (off if off < ub - ua - 1 else 0)
        s = np.arange(first, ub - 1, stride)
        starts.append(s)
        ends.append(np.full(len(s), ub))
    return np.concatenate(starts), np.concatenate(ends)


def make_batch(ev, starts, ends, L=128, mode="all", thr=50):
    n = len(ev["token"])
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    valid = idx < ends[:, None]                      # never read past the train period
    idx = np.minimum(idx, n - 1)
    tok = np.where(valid, ev["token"][idx], 0)       # 0 = PAD (right padding)
    x, y = tok[:, :L], tok[:, 1:]
    ok = valid[:, 1:] & (y > 1)                      # skip PAD and UNK targets
    if mode == "pos":
        ok &= ev["ratio"][idx[:, 1:]] >= thr
    y = np.where(ok, y, -100)
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def eval_samples(ev, period, n_per_user, seed, min_hist=20, L=128):
    rng = np.random.default_rng(seed)
    a, b = ev["bounds"][period]
    idx, cs = [], []
    for us, ua, ub in zip(ev["u_start"], a, b):
        lo = max(ua, us + min_hist)
        if ub <= lo:
            continue
        pick = rng.choice(np.arange(lo, ub), size=min(n_per_user, ub - lo), replace=False)
        idx.append(pick)
        cs.append(np.maximum(us, pick - L))
    return np.concatenate(idx), np.concatenate(cs)


def make_eval_batch(ev, tgt_idx, ctx_start, L=128):
    lens = tgt_idx - ctx_start
    pos = ctx_start[:, None] + np.arange(L)[None, :]
    valid = pos < tgt_idx[:, None]
    x = np.where(valid, ev["token"][np.minimum(pos, len(ev["token"]) - 1)], 0)
    return x, lens, ev["token"][tgt_idx]

def make_rm_batch(ev, starts, ends, L=128):
    """Windows of L consecutive events. Input feedback is shifted by one step (feedback of the PREVIOUS event)."""
    n = len(ev["token"])
    idx = starts[:, None] + np.arange(L)[None, :]
    valid = idx < ends[:, None]
    idx = np.minimum(idx, n - 1)
    tok = np.where(valid, ev["token"][idx], 0)
    r = np.where(valid, ev["ratio"][idx], 0).astype(np.float32) / 100.0
    lk = np.where(valid, ev["like"][idx], 0).astype(np.float32)
    ds = np.where(valid, ev["dislike"][idx], 0).astype(np.float32)
    fb = np.stack([r, (r >= 0.5).astype(np.float32), lk, ds], axis=-1) * valid[..., None]
    fb_in = np.concatenate([np.zeros_like(fb[:, :1]), fb[:, :-1]], axis=1)
    mask = valid & (tok > 1)                      # loss only for in-vocabulary candidates
    t = torch.from_numpy
    return t(tok).long(), t(fb_in).float(), t(r), t(lk), t(ds), t(mask)

def long_targets(ev):
    """Adds ev['g'] (n,4) = [>=1, >=3, >=10 further full listens later in the same session, any listening next calendar day],
    ev['g_ok'] censoring mask, ev['sess_pos'], ev['uidx']."""
    df = pl.read_parquet(ROOT / "data/processed/events.parquet", columns=["uid", "timestamp", "session"])
    uid = df["uid"].to_numpy().astype(np.int64)
    ts = df["timestamp"].to_numpy().astype(np.int64)
    sess = df["session"].to_numpy().astype(np.int64)
    n = len(uid)
    assert n == len(ev["token"])
    new = np.ones(n, dtype=bool)
    new[1:] = (uid[1:] != uid[:-1]) | (sess[1:] != sess[:-1])
    first = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    last = np.concatenate([first[1:], [n]]) - 1
    cs = np.cumsum((ev["ratio"] >= 50).astype(np.int64))
    rem = cs[last[gid]] - cs                         # further full listens in the same session
    day = ts // 86400
    act = pl.read_parquet(ROOT / "data/processed/activity_daily.parquet", columns=["uid", "day"])
    key_act = np.unique(act["uid"].to_numpy().astype(np.int64) * 1000 + act["day"].to_numpy().astype(np.int64))
    key_next = uid * 1000 + day + 1
    pos = np.minimum(np.searchsorted(key_act, key_next), len(key_act) - 1)
    returned = key_act[pos] == key_next
    ev["g"] = np.stack([rem >= 1, rem >= 3, rem >= 10, returned], 1).astype(np.float32)
    ev["g_ok"] = np.stack([np.ones(n, bool)] * 3 + [day + 1 <= 300], 1)    # next-day label needs the next day inside the data
    ev["sess_pos"] = (np.arange(n) - first[gid]).astype(np.float32)
    ev["uidx"] = np.repeat(np.arange(len(ev["u_start"])), ev["u_end"] - ev["u_start"])
    return ev


def make_long_batch(ev, starts, ends, L=128):
    tok, fb_in, *_ = make_rm_batch(ev, starts, ends, L)
    n = len(ev["token"])
    raw = starts[:, None] + np.arange(L)[None, :]
    valid = raw < ends[:, None]
    idx = np.minimum(raw, n - 1)
    ok = ev["g_ok"][idx] & valid[..., None] & (tok.numpy() > 1)[..., None]
    return tok, fb_in, torch.from_numpy(ev["g"][idx]).float(), torch.from_numpy(ok)