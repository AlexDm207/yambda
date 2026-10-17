Set-Location (Split-Path $PSScriptRoot -Parent)

# N: soft novelty-RL and K=1000 reranking (no reward model needed)
foreach ($beta in 1.0,2.0,4.0) {
  $n = "rlnov_beta$beta"
  if (-not (Test-Path "checkpoints\$n\last.pt")) {
    python train/rl.py --name $n --reward novelty --adv_norm std --beta $beta --steps 300 2>&1 | Tee-Object -FilePath "reports\logs\$n.txt"
  }
}
$g = @()
foreach ($beta in 4.0,2.0,1.0,0.4,0.2,0.1) { $g += "RLnov_beta$beta=checkpoints/rlnov_beta$beta/last.pt" }
python scripts/final_eval.py --split val --reward novelty --K 1000 --no_rm --alphas 0 0.5 1 2 3 5 10 20 50 --n_eval 10000 --tag novelty_K1000 --groups $g

# H: sampling budget at beta=0.2 (1024 sequences per step instead of 512)
$jobs = @(@("rl_B16G64_beta0.2", 16, 64), @("rl_B64G16_beta0.2", 64, 16))
foreach ($j in $jobs) {
  $nm, $bs, $gs = $j
  if (-not (Test-Path "checkpoints\$nm\last.pt")) {
    python train/rl.py --name $nm --reward ratio --adv_norm std --beta 0.2 --B $bs --G $gs --steps 300 2>&1 | Tee-Object -FilePath "reports\logs\$nm.txt"
  }
}
$g2 = @("RL_B16G64=checkpoints/rl_B16G64_beta0.2/last.pt", "RL_B64G16=checkpoints/rl_B64G16_beta0.2/last.pt")
python scripts/final_eval.py --split val --n_eval 20000 --tag batch --groups $g2