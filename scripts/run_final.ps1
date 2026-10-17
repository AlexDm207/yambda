param([string]$split = "val", [string]$tag = "main", [int]$n_eval = 6000)
Set-Location (Split-Path $PSScriptRoot -Parent)

$g = @()
foreach ($b in 0.4,0.2,0.1) {
  $g += "RL_beta$b=checkpoints/rlstd_beta$b/last.pt,checkpoints/rlstd_beta${b}_s43/last.pt,checkpoints/rlstd_beta${b}_s44/last.pt"
}
$g += "RL_beta0.05=checkpoints/rl_ratio_b05/last.pt,checkpoints/rlstd_beta0.05_s43/last.pt,checkpoints/rlstd_beta0.05_s44/last.pt"
$g += "Distill_a30=checkpoints/distillfull_a30/last.pt,checkpoints/distillfull_a30_s43/last.pt,checkpoints/distillfull_a30_s44/last.pt"
$g += "Distill_a10=checkpoints/distillfull_a10/last.pt"
$g += "Base_pos=checkpoints/base_pos/best.pt"
if ($split -eq "val") {                       # extra single-seed configurations (explored on val only)
  $g += "RL_debiased_b0.1=checkpoints/rl_debiased_beta0.1/last.pt"
  $g += "RL_debiased_b0.05=checkpoints/rl_debiased_beta0.05/last.pt"
  $g += "RL_seconds_b0.2=checkpoints/rl_seconds_beta0.2/last.pt"
  $g += "RL_sft0_b0.2=checkpoints/rl_sft0_beta0.2/last.pt"
  $g += "RL_sft1_b0.2=checkpoints/rl_sft1_beta0.2/last.pt"
  $g += "Distill_a30_long=checkpoints/distillfull_a30_long/last.pt"
}
python scripts/final_eval.py --split $split --tag $tag --n_eval $n_eval --groups $g