$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$miniMindRoot = Join-Path $repoRoot "model\minimind-v"
$dataPath = Join-Path $miniMindRoot "dataset\zemo_screen_sft.parquet"

if (!(Test-Path $miniMindRoot)) {
  throw "MiniMind-V not found: $miniMindRoot"
}

if (!(Test-Path $dataPath)) {
  throw "Training parquet not found: $dataPath. Run: python scripts\export-minimind-v-sft-data.py"
}

Push-Location (Join-Path $miniMindRoot "trainer")
try {
  python train_sft_vlm.py `
    --data_path "../dataset/zemo_screen_sft.parquet" `
    --save_weight "zemo_screen_sft" `
    --epochs 1 `
    --batch_size 1 `
    --learning_rate 5e-6 `
    --freeze_llm 2 `
    --from_weight "llm" `
    --log_interval 1 `
    --save_interval 20 `
    --num_workers 0 `
    --use_compile 0
}
finally {
  Pop-Location
}

$quantizeScript = Join-Path $repoRoot "scripts\quantize-minimind-v-zemo-screen.py"
python $quantizeScript
