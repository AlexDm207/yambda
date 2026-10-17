Set-Location (Split-Path $PSScriptRoot -Parent)

foreach ($b in 2.0,1.0,0.4,0.2) {
  $n = "rlim_beta$b"
  if (-not (Test-Path "checkpoints\$n\last.pt")) {
    python train/rl.py --name $n --reward itemmean --adv_norm std --beta $b --steps 300 2>&1 | Tee-Object -FilePath "reports\logs\$n.txt"
  }
}
$g = @()
foreach ($b in 2.0,1.0,0.4,0.2) { $g += "RLim_beta$b=checkpoints/rlim_beta$b/last.pt" }
python scripts/final_eval.py --split val --reward itemmean --K 1000 --no_rm --alphas 0 5 10 20 50 100 200 500 1000 --n_eval 10000 --tag itemmean --groups $g