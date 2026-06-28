param(
    [string]$ConverterRoot = "D:\Env\ms\mindspore-lite-2.9.0-win-x64",
    [string]$ModelDir = "E:\CamXAll\ZEMO\uniappx\ZeMo-finance\model\screen_omni_vision_ms",
    [string]$OnnxName = "screen_omni_vision_intent.onnx",
    [string]$OutputName = "screen_omni_vision_intent",
    [int]$FeatureDim = 3072
)

$ErrorActionPreference = "Stop"

$converter = Join-Path $ConverterRoot "tools\converter\converter\converter_lite.exe"
$converterLib = Join-Path $ConverterRoot "tools\converter\lib"
$onnxFile = Join-Path $ModelDir $OnnxName
$outputBase = Join-Path $ModelDir $OutputName

if (!(Test-Path -LiteralPath $converter)) {
    throw "converter_lite.exe not found: $converter"
}

if (!(Test-Path -LiteralPath $onnxFile)) {
    throw "ONNX file not found: $onnxFile. Run scripts\train-screen-omni-vision-ms-intent.py first."
}

$env:Path = "$converterLib;$env:Path"

& $converter `
    --fmk=ONNX `
    --modelFile="$onnxFile" `
    --outputFile="$outputBase" `
    --inputShape="image_features:1,1,1,$FeatureDim" `
    --inputDataFormat=NCHW `
    --optimize=general `
    --fp16=off

if ($LASTEXITCODE -ne 0) {
    throw "converter_lite failed with exit code $LASTEXITCODE"
}

$msFile = "$outputBase.ms"
if (!(Test-Path -LiteralPath $msFile)) {
    throw "converter_lite finished but ms file not found: $msFile"
}

$staticDir = "E:\CamXAll\ZEMO\uniappx\ZeMo-finance\static\models\screen_omni_vision_ms"
New-Item -ItemType Directory -Force -Path $staticDir | Out-Null
Copy-Item -LiteralPath $msFile -Destination (Join-Path $staticDir "screen_omni_vision_intent.ms") -Force
Copy-Item -LiteralPath (Join-Path $ModelDir "label_map.json") -Destination (Join-Path $staticDir "label_map.json") -Force
Copy-Item -LiteralPath (Join-Path $ModelDir "model_config.json") -Destination (Join-Path $staticDir "model_config.json") -Force
Copy-Item -LiteralPath (Join-Path $ModelDir "train_report.json") -Destination (Join-Path $staticDir "train_report.json") -Force
Copy-Item -LiteralPath (Join-Path $ModelDir "README.md") -Destination (Join-Path $staticDir "README.md") -Force

$size = (Get-Item -LiteralPath $msFile).Length
Write-Host "MindSpore Lite screen vision intent model generated: $msFile"
Write-Host "Size bytes: $size"
Write-Host "Static package: $staticDir"
