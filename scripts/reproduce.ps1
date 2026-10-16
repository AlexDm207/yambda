# 0. environment (see README): python -m venv .venv ; pip install torch --index-url .../cu124 ; pip install -r requirements.txt
python data/download.py                    # Yambda-50M, multi_event.parquet
python data/prepare.py                     # temporal split, vocabulary (train only), leak checks
python scripts/evaluate.py --split val     # popularity and history baselines
python train/sft.py --mode all --name base_all --epochs 4
python train/sft.py --mode pos --name base_pos --epochs 4
python train/train_rm.py --name rm_a --epochs 3
python train/train_rm.py --name rm_b --seed 7 --n_layer 3 --n_embd 192 --epochs 3
python train/train_long.py --name rm_long --epochs 3
foreach ($s in 42,43,44) { foreach ($b in 0.4,0.2,0.1,0.05) {
  python train/rl.py --name "rlstd_beta${b}_s$s" --reward ratio --adv_norm std --beta $b --steps 300 --seed $s } }
foreach ($s in 42,43,44) { python train/distill.py --name "distillfull_a30_s$s" --reward ratio --alpha 30 --target full --steps 1500 --seed $s }
python train/distill.py --name distillfull_a10 --reward ratio --alpha 10 --target full --steps 1500
.\scripts\run_final.ps1                              # val
.\scripts\run_final.ps1 -split test -tag test        # test, run once
python scripts/make_figures.py ; python scripts/make_tables.py