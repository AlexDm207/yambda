from huggingface_hub import HfApi, hf_hub_download

REPO = "yandex/yambda"

# 1) Посмотреть, что вообще есть в репозитории и сколько это весит
api = HfApi()
info = api.dataset_info(REPO, files_metadata=True)
for s in info.siblings:
    if s.rfilename.startswith(("flat/50m", "embeddings", "album", "artist")):
        print(f"{(s.size or 0)/1e6:10.1f} MB  {s.rfilename}")

# 2) Скачать только объединённый файл событий
path = hf_hub_download(
    repo_id=REPO, repo_type="dataset",
    filename="flat/50m/multi_event.parquet",
    local_dir="data/raw",
)
print("saved to:", path)
