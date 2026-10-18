Set-Location (Split-Path $PSScriptRoot -Parent)

# groups "name=ckpt" only for checkpoints that exist (a failed training run must not break the evaluation)
function Groups($prefix, $dir, $betas) {
  $out = @()
  foreach ($b in $betas) {
    $p = "checkpoints/$dir$b/last.pt"
    if (Test-Path $p) { $out += "$prefix$b=$p" } else { Write-Warning "missing $p, skipped" }
  }
  return ,$out
}

# B: reward without a reward model (novelty = log10 popularity rank)
foreach ($b in 0.4,0.2,0.1) {
  $n = "rlnov_beta$b"
  if (-not (Test-Path "checkpoints\$n\last.pt")) {
    python train/rl.py --name $n --reward novelty --adv_norm std --beta $b --steps 300 2>&1 | Tee-Object -FilePath "reports\logs\$n.txt"
  }
}
$g = Groups "RLnov_beta" "rlnov_beta" @(0.4,0.2,0.1)
if ($g.Count) { python scripts/final_eval.py --split val --reward novelty --alphas 0 0.5 1 2 5 10 20 50 --n_eval 20000 --tag novelty --groups $g }

# F: reward without the item main effect (what is left is the context-specific part)
foreach ($b in 0.2,0.05) {
  $n = "rl_itemdeb_beta$b"
  if (-not (Test-Path "checkpoints\$n\last.pt")) {
    python train/rl.py --name $n --reward itemdebiased --adv_norm std --beta $b --steps 300 2>&1 | Tee-Object -FilePath "reports\logs\$n.txt"
  }
}
$g2 = Groups "RLitemdeb_beta" "rl_itemdeb_beta" @(0.2,0.05)
if ($g2.Count) { python scripts/final_eval.py --split val --n_eval 20000 --tag itemdeb --groups $g2 }
