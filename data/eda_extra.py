import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

RAW = "data/raw/flat/50m/multi_event.parquet"
DAY = 86400
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)

lf = pl.scan_parquet(RAW)
listens = lf.filter(pl.col("event_type") == "listen")

print("=== 0. Единицы времени ===")
print("дней (если секунды):", lf.select(pl.col("timestamp").max()).collect().item() / DAY)
print("доля timestamp кратных 5:", lf.select((pl.col("timestamp") % 5 == 0).mean()).collect().item())

print("\n=== 1. Полные квантили ===")
qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
for col in ["played_ratio_pct", "track_length_seconds"]:
    vals = listens.select([pl.col(col).quantile(q).alias(str(q)) for q in qs]).collect().row(0)
    print(col, dict(zip(qs, vals)))

print("\n=== 2. organic vs non-organic ===")
print(listens.group_by("is_organic").agg(
    pl.len().alias("n"),
    (pl.col("played_ratio_pct") >= 50).mean().alias("share_ge50"),
    (pl.col("played_ratio_pct") <= 10).mean().alias("share_le10"),
    pl.col("played_ratio_pct").clip(0, 100).mean().alias("mean_ratio_clip"),
).sort("is_organic").collect())

print("\n=== 3. Доля прослушивания по длине трека (минуты) ===")
print(listens.group_by((pl.col("track_length_seconds") // 60).clip(0, 8).alias("len_min")).agg(
    pl.len().alias("n"),
    pl.col("played_ratio_pct").clip(0, 100).mean().alias("mean_ratio"),
    (pl.col("played_ratio_pct") >= 50).mean().alias("share_ge50"),
).sort("len_min").collect())

print("\n=== 4. Лайки: сколько реально остаётся ===")
ev = lf.filter((pl.col("event_type") == "like") | (pl.col("event_type") == "unlike"))
pair = ev.group_by("uid", "item_id").agg(
    (pl.col("event_type") == "like").sum().alias("n_like"),
    (pl.col("event_type") == "unlike").sum().alias("n_unlike"),
)
print(pair.select(
    pl.len().alias("pairs"),
    (pl.col("n_like") > pl.col("n_unlike")).sum().alias("net_liked"),
    (pl.col("n_unlike") >= pl.col("n_like")).sum().alias("fully_unliked"),
).collect())

print("\n=== 5. Активность по дням ===")
per_day = (listens.group_by((pl.col("timestamp") // DAY).alias("day"))
           .agg(pl.len().alias("n"), pl.col("uid").n_unique().alias("active_users"))
           .sort("day").collect())
print(per_day.head(5)); print(per_day.tail(5))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(per_day["day"], per_day["n"]); ax[0].set_title("Прослушиваний в день")
ax[1].plot(per_day["day"], per_day["active_users"]); ax[1].set_title("Активных пользователей в день")
plt.tight_layout(); plt.savefig(FIG / "eda_per_day.png", dpi=120); plt.close()

h = (listens.group_by(pl.col("played_ratio_pct").clip(0, 100).alias("r"))
     .agg(pl.len().alias("n")).sort("r").collect())
plt.figure(figsize=(7, 3.5)); plt.bar(h["r"], h["n"]); plt.title("Распределение played_ratio (обрезано до 100)")
plt.ylabel("число прослушиваний")
plt.tight_layout(); plt.savefig(FIG / "eda_ratio_hist.png", dpi=120); plt.close()

print("\n=== 6. Доля повторов (organic) ===")
org = (listens.filter(pl.col("is_organic") == 1)
       .select("uid", "timestamp", "item_id").sort(["uid", "timestamp"]))
print(org.with_columns(pl.col("item_id").is_first_distinct().over("uid").alias("first"))
      .select((1 - pl.col("first").cast(pl.Float64).mean()).alias("repeat_share")).collect())

print("\n=== 7. Сессии (разрыв > 30 мин = 1800 с) ===")
sess = (listens.select("uid", "timestamp").sort(["uid", "timestamp"])
        .with_columns((pl.col("timestamp").diff().over("uid").fill_null(10**9) > 1800).alias("new"))
        .with_columns(pl.col("new").cum_sum().over("uid").alias("sid"))
        .group_by("uid", "sid").agg(pl.len().alias("n")))
print(sess.select(pl.len().alias("n_sessions"),
                  pl.col("n").median().alias("median_len"),
                  pl.col("n").quantile(0.9).alias("p90_len")).collect())

print("\n=== 8. Покрытие словаря на train-периоде (первые 270 дней) ===")
TR = 270 * DAY
org_l = listens.filter(pl.col("is_organic") == 1)
cnt = (org_l.filter(pl.col("timestamp") < TR)
       .group_by("item_id").agg(pl.len().alias("c")).sort("c", descending=True).collect())
total = cnt["c"].sum()
for N in [20_000, 30_000, 50_000, 100_000]:
    print(f"top-{N}: {cnt['c'][:N].sum() / total:.1%} train-событий")
for N in [30_000, 50_000, 100_000]:
    top = cnt["item_id"][:N]
    share = (org_l.filter(pl.col("timestamp") >= TR)
             .select(pl.col("item_id").is_in(top).mean()).collect().item())
    print(f"top-{N}: {share:.1%} будущих событий (после дня 270) внутри словаря")