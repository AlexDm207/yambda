import polars as pl

lf = pl.scan_parquet("data/raw/flat/50m/multi_event.parquet")
print(lf.collect_schema(), "\n")

# 1. Сколько событий каждого типа
print(lf.group_by("event_type").agg(pl.len().alias("n")).collect(), "\n")

# 2. Общая статистика
print(lf.select(
    pl.len().alias("rows"),
    pl.col("uid").n_unique().alias("users"),
    pl.col("item_id").n_unique().alias("items"),
    pl.col("timestamp").min().alias("ts_min"),
    pl.col("timestamp").max().alias("ts_max"),
    pl.col("is_organic").mean().alias("organic_share"),
).collect(), "\n")

# 3. Распределение доли прослушивания и длительности (только listen)
listens = lf.filter(pl.col("event_type") == "listen")
qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
print(listens.select(
    [pl.col("played_ratio_pct").quantile(q).alias(f"ratio_q{q}") for q in qs]
    + [pl.col("track_length_seconds").quantile(q).alias(f"len_q{q}") for q in qs]
).collect().transpose(include_header=True), "\n")
print("доля прослушиваний с ratio>=50:",
      listens.select((pl.col("played_ratio_pct") >= 50).mean()).collect().item())
print("доля прослушиваний с ratio>100:",
      listens.select((pl.col("played_ratio_pct") > 100).mean()).collect().item())

# 4. Длины историй пользователей
lens = lf.group_by("uid").agg(pl.len().alias("n")).collect()["n"]
print("длина истории: ", lens.quantile(0.1), lens.quantile(0.5), lens.quantile(0.9), lens.max(), "\n")

# 5. Покрытие словаря: какую долю событий закрывают топ-N треков
cnt = lf.group_by("item_id").agg(pl.len().alias("c")).sort("c", descending=True).collect()
total = cnt["c"].sum()
for N in [10_000, 50_000, 100_000, 200_000]:
    print(f"top-{N}: {cnt['c'][:N].sum()/total:.1%} событий")

# 6. Шаг по времени между событиями одного пользователя (чтобы понять единицы timestamp)
gaps = (lf.sort(["uid", "timestamp"])
          .with_columns(pl.col("timestamp").diff().over("uid").alias("gap"))
          .select(pl.col("gap").quantile(0.5).alias("median_gap"),
                  pl.col("gap").quantile(0.99).alias("p99_gap"))
          .collect())
print(gaps)
