"""Stage 1: events table, temporal split, vocabulary (train only), leak checks."""
import json
from pathlib import Path

import polars as pl
import yaml

DAY = 86_400
cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
V = cfg["data"]["vocab_size"]
SESSION_GAP = cfg["data"]["session_gap_sec"]
LABEL_WINDOW = cfg["data"]["label_window_sec"]
S = cfg["split"]
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

lf = pl.scan_parquet(cfg["data"]["raw_path"])

# ---------- 1. Listens ----------
all_listens = lf.filter(pl.col("event_type") == "listen").select(
    "uid", "timestamp", "item_id", "is_organic",
    pl.col("played_ratio_pct").clip(0, 100).cast(pl.UInt8).alias("ratio"),
    pl.col("track_length_seconds").alias("length"),
)

# Daily activity from ALL listens (organic + recommended): for return/retention signals later
daily = (
    all_listens.group_by("uid", (pl.col("timestamp") // DAY).alias("day"))
    .agg(
        pl.len().alias("n_listens_all"),
        (pl.col("is_organic") == 1).sum().alias("n_organic"),
        (pl.col("ratio").cast(pl.Float32) / 100 * pl.col("length")).sum().alias("play_seconds"),
    )
    .sort("uid", "day")
    .collect()
)
daily.write_parquet(OUT / "activity_daily.parquet", compression="zstd")
print("activity_daily:", daily.shape)

org = (
    all_listens.filter(pl.col("is_organic") == 1)
    .drop("is_organic")
    .collect()
    .sort("timestamp", maintain_order=True)
)
print("organic listens:", org.height)

# ---------- 2. Like / dislike labels: event on the same (uid, item) within LABEL_WINDOW after listen start ----------
def feedback(kind: str, name: str) -> pl.DataFrame:
    return (
        lf.filter(pl.col("event_type") == kind)
        .select("uid", "item_id", "timestamp", pl.col("timestamp").alias(f"{name}_ts"))
        .collect()
        .sort("timestamp", maintain_order=True)
    )

for kind in ["like", "dislike"]:
    fb = feedback(kind, kind)
    org = org.join_asof(
        fb, on="timestamp", by=["uid", "item_id"],
        strategy="forward", tolerance=LABEL_WINDOW,
    )
    org = org.with_columns(pl.col(f"{kind}_ts").is_not_null().cast(pl.UInt8).alias(kind)).drop(f"{kind}_ts")

# ---------- 3. Period assignment ----------
t = pl.col("timestamp")
b = {k: v * DAY for k, v in S.items()}
org = org.with_columns(
    pl.when(t < b["train_end_day"]).then(pl.lit("train"))
    .when(t < b["val_start_day"]).then(pl.lit("gap"))
    .when(t < b["val_end_day"]).then(pl.lit("val"))
    .when(t < b["test_start_day"]).then(pl.lit("gap"))
    .otherwise(pl.lit("test"))
    .cast(pl.Enum(["train", "gap", "val", "test"]))
    .alias("period")
)

# ---------- 4. Vocabulary from TRAIN only ----------
vocab = (
    org.filter(pl.col("period") == "train")
    .group_by("item_id").agg(pl.len().alias("train_count"))
    .sort(["train_count", "item_id"], descending=[True, False])
    .head(V)
    .with_row_index("rank")
    .with_columns((pl.col("rank") + 2).cast(pl.UInt32).alias("token"))
    .select("item_id", "token", "train_count")
)
vocab.write_parquet(OUT / "vocab.parquet")

org = (
    org.join(vocab.select("item_id", "token"), on="item_id", how="left")
    .with_columns(pl.col("token").fill_null(1).cast(pl.UInt32))  # 1 = UNK
    .sort(["uid", "timestamp"], maintain_order=True)
)

# ---------- 5. Sessions and position inside user history ----------
org = (
    org.with_columns(
        (pl.col("timestamp").diff().over("uid").fill_null(SESSION_GAP + 1) > SESSION_GAP)
        .cast(pl.UInt32).alias("new_sess")
    )
    .with_columns(
        pl.col("new_sess").cum_sum().over("uid").alias("session"),
        pl.int_range(pl.len(), dtype=pl.UInt32).over("uid").alias("pos"),
    )
    .drop("new_sess")
)
org.write_parquet(OUT / "events.parquet", compression="zstd")

# ---------- 6. Statistics ----------
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
print("\n=== per period ===")
print(
    org.group_by("period").agg(
        pl.len().alias("n"),
        pl.col("uid").n_unique().alias("users"),
        (pl.col("token") > 1).mean().alias("in_vocab"),
        (pl.col("ratio") >= 50).mean().alias("pos_rate"),
        pl.col("ratio").cast(pl.Float32).mean().alias("mean_ratio"),
        pl.col("like").mean().alias("like_rate"),
        pl.col("dislike").mean().alias("dislike_rate"),
        pl.col("length").mean().alias("mean_len"),
    ).sort("period")
)

train_items = org.filter(pl.col("period") == "train").select("item_id").unique()
for p in ["val", "test"]:
    sub = org.filter(pl.col("period") == p)
    cold = sub.join(train_items, on="item_id", how="anti").height / sub.height
    print(f"{p}: доля событий с треком, которого не было в train: {cold:.1%}")

# users that can be evaluated: >=20 events before the period start and >=1 event inside it
for p, start in [("val", S["val_start_day"]), ("test", S["test_start_day"])]:
    hist = org.filter(pl.col("timestamp") < start * DAY).group_by("uid").agg(pl.len().alias("h"))
    inside = org.filter(pl.col("period") == p).group_by("uid").agg(pl.len().alias("n"))
    ok = hist.join(inside, on="uid").filter(pl.col("h") >= 20)
    print(f"{p}: пользователей для оценки: {ok.height}, событий в периоде у них: {ok['n'].sum()}")

# ---------- 7. Leak checks ----------
def rng(p):
    s = org.filter(pl.col("period") == p)["timestamp"]
    return s.min(), s.max()

assert rng("train")[1] < b["train_end_day"]
assert rng("val")[0] >= b["val_start_day"] and rng("val")[1] < b["val_end_day"]
assert rng("test")[0] >= b["test_start_day"]
assert vocab.height == V and vocab["token"].min() == 2
assert org.select(pl.col("timestamp").diff().over("uid").min()).item() >= 0, "events not sorted by time"
print("\nLeak checks passed.")

json.dump(
    {"vocab_size": V, "pad": 0, "unk": 1, "split_days": S, "events": org.height},
    open(OUT / "meta.json", "w", encoding="utf-8"), indent=2,
)