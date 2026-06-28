param(
  [string]$FlatcPath = "flatc",
  [string]$NpxPath = "npx"
)

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$schemaPath = Join-Path $repoRoot 'services\sync\fbs\sync-envelope.fbs'
$androidOut = Join-Path $repoRoot 'uni_modules\zemo-flatbuffers\utssdk\app-android\src'
$harmonyTsOut = Join-Path $repoRoot 'uni_modules\zemo-flatbuffers\utssdk\app-harmony\src\generated-ts'
$harmonyJsOut = Join-Path $repoRoot 'uni_modules\zemo-flatbuffers\utssdk\app-harmony\src\generated'
$harmonyRuntimeEntry = Join-Path $repoRoot 'uni_modules\zemo-flatbuffers\utssdk\app-harmony\src\vendor\flatbuffers\mjs\flatbuffers.js'

if (-not (Test-Path $schemaPath)) {
    throw "FlatBuffers schema not found: $schemaPath"
}

function Get-NormalizedRelativeImportPath {
  param(
    [string]$FromFile,
    [string]$ToFile
  )

  Push-Location (Split-Path $FromFile -Parent)
  try {
    $relativePath = Resolve-Path -Path $ToFile -Relative
  } finally {
    Pop-Location
  }
  $relativePath = $relativePath -replace '\\', '/'
  if (-not $relativePath.StartsWith('.')) {
    $relativePath = "./$relativePath"
  }
  return $relativePath
}

$flatc = Get-Command $FlatcPath -ErrorAction SilentlyContinue
if ($null -eq $flatc) {
    throw @"
flatc was not found.

Install the official FlatBuffers compiler first, then rerun this script.
Expected command:
  flatc --version

This script generates:
  - Java classes for Android official runtime
  - TypeScript sources and transpiled JavaScript for Harmony official runtime integration

Additional Harmony requirements:
  - Node.js with npx available
  - Vendored FlatBuffers JS runtime under:
    uni_modules/zemo-flatbuffers/utssdk/app-harmony/src/vendor/flatbuffers
"@
}

if (-not (Test-Path $harmonyRuntimeEntry)) {
    throw @"
Vendored Harmony FlatBuffers runtime not found:
  $harmonyRuntimeEntry

Vendor the official JS runtime into:
  uni_modules/zemo-flatbuffers/utssdk/app-harmony/src/vendor/flatbuffers
then rerun this script.
"@
}

$npx = Get-Command $NpxPath -ErrorAction SilentlyContinue
if ($null -eq $npx) {
    throw @"
npx was not found.

Install Node.js so this script can transpile generated TypeScript to JavaScript
for the Harmony UTS mixed source path.
"@
}

New-Item -ItemType Directory -Force -Path $androidOut | Out-Null
foreach ($path in @($harmonyTsOut, $harmonyJsOut)) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force -Path $path
    }
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

& $FlatcPath --java -o $androidOut $schemaPath
& $FlatcPath --ts --gen-object-api -o $harmonyTsOut $schemaPath

$harmonyTsFiles = Get-ChildItem -Path $harmonyTsOut -Recurse -Filter '*.ts' | Sort-Object FullName
foreach ($tsFile in $harmonyTsFiles) {
    $content = Get-Content -Raw -Path $tsFile.FullName
    if ($content.Contains("from 'flatbuffers';")) {
        $runtimeImportPath = Get-NormalizedRelativeImportPath -FromFile $tsFile.FullName -ToFile $harmonyRuntimeEntry
        $updatedContent = $content.Replace("from 'flatbuffers';", "from '$runtimeImportPath';")
        Set-Content -Path $tsFile.FullName -Value $updatedContent -NoNewline
    }
}

& $NpxPath --yes -p typescript tsc `
    --target ES2020 `
    --module ES2020 `
  --moduleResolution bundler `
    --skipLibCheck `
    --rootDir $harmonyTsOut `
    --outDir $harmonyJsOut `
    @($harmonyTsFiles | ForEach-Object { $_.FullName })

if ($LASTEXITCODE -ne 0) {
    throw "Harmony TypeScript transpilation failed."
}

Write-Host "Generated Android Java sources into: $androidOut"
Write-Host "Generated Harmony TypeScript sources into: $harmonyTsOut"
Write-Host "Generated Harmony JavaScript sources into: $harmonyJsOut"
