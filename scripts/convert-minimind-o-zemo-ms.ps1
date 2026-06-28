param(
  [string]$ConverterRoot = "D:\Env\ms\mindspore-lite-2.9.0-win-x64",
  [string]$PythonExe = "",
  [int]$SeqLen = 192
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$miniRoot = Join-Path $repoRoot "model\minimind-o"
$exportScript = Join-Path $repoRoot "scripts\export-minimind-o-zemo-onnx.py"
$outDir = Join-Path $miniRoot "mindspore_lite"
$onnxFile = Join-Path $outDir "zemo_screen_minimind_o_prefill_s$SeqLen.onnx"
$msPrefix = Join-Path $outDir "zemo_screen_minimind_o_prefill_s$SeqLen"
$msFile = "$msPrefix.ms"
$manifestFile = Join-Path $outDir "zemo_screen_minimind_o_prefill_s$SeqLen.manifest.json"
$converter = Join-Path $ConverterRoot "tools\converter\converter\converter_lite.exe"
$converterLib = Join-Path $ConverterRoot "tools\converter\lib"
$staticDir = Join-Path $repoRoot "static\models\minimind_o_ms"
$rawDir = Join-Path $repoRoot "harmony-configs\entry\src\main\resources\rawfile\static\models\minimind_o_ms"
$importDir = Join-Path $repoRoot "model\minimind_o_ms"

if ($PythonExe -eq "") {
  $venvPython = Join-Path $miniRoot ".venv\Scripts\python.exe"
  $PythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
}

if (!(Test-Path -LiteralPath $converter)) {
  throw "converter_lite.exe not found: $converter"
}

New-Item -ItemType Directory -Force -Path $outDir, $staticDir, $rawDir, $importDir | Out-Null

& $PythonExe $exportScript --seq-len $SeqLen --output $onnxFile --device cpu
if ($LASTEXITCODE -ne 0) {
  throw "ONNX export failed with exit code $LASTEXITCODE"
}

$env:Path = "$converterLib;$env:Path"
& $converter `
  --fmk=ONNX `
  --modelFile=$onnxFile `
  --outputFile=$msPrefix `
  --optimize=general `
  --infer=false
if ($LASTEXITCODE -ne 0) {
  throw "converter_lite failed with exit code $LASTEXITCODE"
}
if (!(Test-Path -LiteralPath $msFile)) {
  throw "converter_lite finished but ms file not found: $msFile"
}

& $PythonExe -c "import json, pathlib, sys; src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); data=json.loads(src.read_text(encoding='utf-8')); vocab=data['model']['vocab']; arr=['']*(max(vocab.values())+1); [arr.__setitem__(int(i), t) for t,i in vocab.items()]; out.write_text(json.dumps(arr, ensure_ascii=False, separators=(',',':')), encoding='utf-8')" `
  (Join-Path $miniRoot "zemo-screen-omni-final\tokenizer.json") `
  (Join-Path $staticDir "id_to_token.json")
if ($LASTEXITCODE -ne 0) {
  throw "id_to_token export failed with exit code $LASTEXITCODE"
}

Copy-Item -LiteralPath $manifestFile -Destination (Join-Path $staticDir "manifest.json") -Force
Copy-Item -LiteralPath (Join-Path $miniRoot "zemo-screen-omni-final\tokenizer.json") -Destination (Join-Path $staticDir "tokenizer.json") -Force
Copy-Item -LiteralPath $msFile -Destination (Join-Path $importDir "zemo_screen_minimind_o_prefill_s$SeqLen.ms") -Force
Copy-Item -LiteralPath $manifestFile -Destination (Join-Path $importDir "zemo_screen_minimind_o_prefill_s$SeqLen.manifest.json") -Force
Copy-Item -LiteralPath $manifestFile -Destination (Join-Path $importDir "manifest.json") -Force
Copy-Item -LiteralPath (Join-Path $staticDir "id_to_token.json") -Destination (Join-Path $importDir "id_to_token.json") -Force
Copy-Item -LiteralPath (Join-Path $staticDir "tokenizer.json") -Destination (Join-Path $importDir "tokenizer.json") -Force

Remove-Item -LiteralPath (Join-Path $staticDir "zemo_screen_minimind_o_prefill_s$SeqLen.ms") -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $rawDir "zemo_screen_minimind_o_prefill_s$SeqLen.ms") -Force -ErrorAction SilentlyContinue

Copy-Item -LiteralPath (Join-Path $staticDir "manifest.json") -Destination (Join-Path $rawDir "manifest.json") -Force
Copy-Item -LiteralPath (Join-Path $staticDir "id_to_token.json") -Destination (Join-Path $rawDir "id_to_token.json") -Force
Copy-Item -LiteralPath (Join-Path $staticDir "tokenizer.json") -Destination (Join-Path $rawDir "tokenizer.json") -Force

Write-Host "MiniMind-O MindSpore Lite exported:"
Write-Host "  $msFile"
Write-Host "  Import copy: $(Join-Path $importDir "zemo_screen_minimind_o_prefill_s$SeqLen.ms")"
Write-Host "  App package resources updated with manifest/tokenizer only."
Write-Host "  Import the .ms file in app settings after installing the HAP."
