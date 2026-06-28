$ErrorActionPreference = "Stop"

param(
  [string]$ImageDir = "../data/images/guiact_smartphone_test",
  [int]$MaxNewTokens = 64,
  [int]$Quantized = 1
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$miniMindRoot = Join-Path $repoRoot "model\minimind-v"

Push-Location $miniMindRoot
try {
  python eval_vlm.py `
    --load_from model `
    --weight zemo_screen_sft `
    --quantized $Quantized `
    --hidden_size 768 `
    --num_hidden_layers 8 `
    --use_moe 0 `
    --max_new_tokens $MaxNewTokens `
    --temperature 0.1 `
    --top_p 0.9 `
    --image_dir $ImageDir `
    --show_speed 1 `
    --device cpu
}
finally {
  Pop-Location
}
