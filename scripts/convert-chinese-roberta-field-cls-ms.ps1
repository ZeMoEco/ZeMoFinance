param(
    [string]$ConverterRoot = "D:\Env\ms\mindspore-lite-2.9.0-win-x64",
    [string]$ModelDir = "E:\CamXAll\ZEMO\Data\model\finance_field_cls_model",
    [string]$OnnxName = "finance_field_cls.onnx",
    [string]$OutputName = "finance_field_cls",
    [int]$SeqLen = 128,
    [int]$FeatureDim = 4096,
    [int]$FeatureSide = 64
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
    throw "ONNX file not found: $onnxFile. Run scripts\export-chinese-roberta-field-cls-onnx.py first."
}

$env:Path = "$converterLib;$env:Path"

& $converter `
    --fmk=ONNX `
    --modelFile="$onnxFile" `
    --outputFile="$outputBase" `
    --inputShape="features:1,1,$FeatureSide,$FeatureSide" `
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

Write-Host "MindSpore Lite model generated: $msFile"

$packageDir = Join-Path $ModelDir "slm"
$packageMsFile = Join-Path $packageDir "finance_field_cls.ms"
$packageVocabFile = Join-Path $packageDir "vocab.txt"
$packageLabelsFile = Join-Path $packageDir "label_map.json"

New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
Copy-Item -LiteralPath $msFile -Destination $packageMsFile -Force

$rootVocabFile = Join-Path $ModelDir "vocab.txt"
$rootLabelsFile = Join-Path $ModelDir "label_map.json"

if (!(Test-Path -LiteralPath $packageVocabFile) -and (Test-Path -LiteralPath $rootVocabFile)) {
    Copy-Item -LiteralPath $rootVocabFile -Destination $packageVocabFile -Force
}

if (!(Test-Path -LiteralPath $packageLabelsFile) -and (Test-Path -LiteralPath $rootLabelsFile)) {
    Copy-Item -LiteralPath $rootLabelsFile -Destination $packageLabelsFile -Force
}

if (!(Test-Path -LiteralPath $packageVocabFile)) {
    throw "SLM package missing vocab.txt: $packageVocabFile"
}

if (!(Test-Path -LiteralPath $packageLabelsFile)) {
    throw "SLM package missing label_map.json: $packageLabelsFile"
}

Write-Host "SLM import package updated:"
Write-Host "  $packageMsFile"
Write-Host "  $packageVocabFile"
Write-Host "  $packageLabelsFile"

Write-Host "SLM static resource sync skipped. Import the three files from the slm package in the app."
