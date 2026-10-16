Set-Location (Split-Path $PSScriptRoot -Parent)

function Run-RL($name, $reward, $beta, $sft) {
    if (-not (Test-Path "checkpoints\$name\last.pt")) {
        python train/rl.py --name $name --reward $reward --adv_norm std --beta $beta --sft $sft --steps 300 2>&1 |
            Tee-Object -FilePath "reports\logs\$name.txt"
    } else { Write-Host "skip training: $name" }
    Eval-One $name
}

function Eval-One($name) {
    if (-not (Test-Path "reports\results\val_reward_${name}_last_ratio_${name}.json")) {
        python scripts/eval_reward.py --ckpt "checkpoints/$name/last.pt" --reward ratio --tag "_$name"
    } else { Write-Host "skip eval: $name" }
}

Run-RL "rl_debiased_beta0.1"  "debiased" 0.1  0.1
Run-RL "rl_debiased_beta0.05" "debiased" 0.05 0.1
Run-RL "rl_seconds_beta0.2"   "seconds"  0.2  0.1
Run-RL "rl_sft0_beta0.2"      "ratio"    0.2  0
Run-RL "rl_sft1_beta0.2"      "ratio"    0.2  1

if (-not (Test-Path "reports\results\val_reward_distillfull_a30_long_last_ratio_a30_long.json")) {
    python scripts/eval_reward.py --ckpt checkpoints/distillfull_a30_long/last.pt --reward ratio --tag _a30_long
}
Write-Host "extra runs done"