param(
  [ValidateSet("screen", "mini", "full", "all")]
  [string]$Stage = "screen",
  [string]$CudaVisibleDevices = "0",
  [int]$NprocPerNode = 1,
  [int]$MasterPort = 29560,
  [switch]$UseMoe,
  [switch]$UseWandb,
  [switch]$InstallRequirements,
  [switch]$DownloadModels,
  [switch]$DownloadOfficialData,
  [switch]$SkipClone,
  [int]$ScreenEpochs = 2,
  [int]$ScreenBatchSize = 8,
  [string]$ScreenFromWeight = "sft_omni",
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$miniMindORoot = Join-Path $repoRoot "model\minimind-o"
$datasetDir = Join-Path $miniMindORoot "dataset"
$trainerDir = Join-Path $miniMindORoot "trainer"
$exportScript = Join-Path $repoRoot "scripts\export-minimind-o-zemo-screen-data.py"

if ($PythonExe -eq "") {
  $venvPython = Join-Path $miniMindORoot ".venv\Scripts\python.exe"
  $PythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
}

function Invoke-Py {
  & $PythonExe @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

function Invoke-ModelScope {
  $venvModelScope = Join-Path $miniMindORoot ".venv\Scripts\modelscope.exe"
  if (Test-Path $venvModelScope) {
    & $venvModelScope @args
  } else {
    modelscope @args
  }
}

function Ensure-MiniMindO {
  if (Test-Path $miniMindORoot) { return }
  if ($SkipClone) {
    throw "MiniMind-O not found: $miniMindORoot"
  }
  git clone --depth 1 https://github.com/jingyaogong/minimind-o $miniMindORoot
}

function Invoke-PythonInstall {
  if (!$InstallRequirements) { return }
  Push-Location $miniMindORoot
  try {
    Invoke-Py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  } finally {
    Pop-Location
  }
}

function Invoke-ModelDownload {
  if (!$DownloadModels) { return }
  Push-Location $miniMindORoot
  try {
    Invoke-ModelScope download --model gongjy/SenseVoiceSmall --local_dir .\model\SenseVoiceSmall
    Invoke-ModelScope download --model gongjy/siglip2-base-p32-256-ve --local_dir .\model\siglip2-base-p32-256-ve
    Invoke-ModelScope download --model gongjy/mimi --local_dir .\model\mimi
    Invoke-ModelScope download --model gongjy/campplus --local_dir .\model\campplus
    Invoke-ModelScope download --model gongjy/minimind-3o-pytorch llm_768.pth --local_dir .\out
  } finally {
    Pop-Location
  }
}

function Invoke-OfficialDataDownload {
  if (!$DownloadOfficialData) { return }
  New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
  $files = @()
  if ($Stage -eq "mini") {
    $files += @("sft_t2a_mini.parquet", "sft_a2a_mini.parquet")
  }
  if ($Stage -eq "full" -or $Stage -eq "all") {
    $files += @("sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet")
  }
  foreach ($file in $files) {
    Invoke-Py -c "from huggingface_hub import hf_hub_download; import sys; hf_hub_download(repo_id='jingyaogong/minimind-o_dataset', repo_type='dataset', filename=sys.argv[1], local_dir=sys.argv[2])" $file $datasetDir
  }
}

function Export-ZeMoScreenData {
  Invoke-Py $exportScript --copy-to-minimind-o --minimind-o-root $miniMindORoot
}

function Invoke-OmniTrain {
  param(
    [string]$LearningRate,
    [string]$DataPath,
    [int]$Epochs,
    [int]$BatchSize,
    [string]$FromWeight,
    [string]$SaveWeight,
    [int]$MaxSeqLen,
    [string]$Mode = "all",
    [int]$UseCompile = 0
  )
  $env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
  $env:USE_LIBUV = "0"
  $trainArgs = @(
    "--learning_rate", $LearningRate,
    "--data_path", $DataPath,
    "--epochs", $Epochs.ToString(),
    "--batch_size", $BatchSize.ToString(),
    "--from_weight", $FromWeight,
    "--save_weight", $SaveWeight,
    "--max_seq_len", $MaxSeqLen.ToString(),
    "--mode", $Mode,
    "--use_compile", $UseCompile.ToString(),
    "--use_moe", ($(if ($UseMoe) { "1" } else { "0" })),
    "--num_workers", "0"
  )
  if ($UseWandb) { $trainArgs += "--use_wandb" }

  if ($NprocPerNode -le 1) {
    Invoke-Py "train_sft_omni.py" @trainArgs
    return
  }

  $argsList = @(
    "--master_port", $MasterPort.ToString(),
    "--nproc_per_node", $NprocPerNode.ToString(),
    "train_sft_omni.py"
  ) + $trainArgs
  Invoke-Py -m torch.distributed.run @argsList
}

function Invoke-MiniPipeline {
  Invoke-OmniTrain -LearningRate "5e-4" -DataPath "..\dataset\sft_t2a_mini.parquet" -Epochs 1 -BatchSize 40 -FromWeight "llm" -SaveWeight "sft_zero" -MaxSeqLen 512 -Mode "all" -UseCompile 1
  Invoke-OmniTrain -LearningRate "5e-4" -DataPath "..\dataset\sft_a2a_mini.parquet" -Epochs 1 -BatchSize 40 -FromWeight "sft_zero" -SaveWeight "sft_zero" -MaxSeqLen 640 -Mode "audio_proj" -UseCompile 0
  Invoke-OmniTrain -LearningRate "2e-5" -DataPath "..\dataset\sft_a2a_mini.parquet" -Epochs 1 -BatchSize 16 -FromWeight "sft_zero" -SaveWeight "sft_zero" -MaxSeqLen 768 -Mode "all" -UseCompile 0
}

function Invoke-FullPipeline {
  Invoke-OmniTrain -LearningRate "5e-4" -DataPath "..\dataset\sft_t2a.parquet" -Epochs 6 -BatchSize 32 -FromWeight "llm" -SaveWeight "sft_omni" -MaxSeqLen 512 -Mode "all" -UseCompile 1
  Invoke-OmniTrain -LearningRate "5e-4" -DataPath "..\dataset\sft_a2a.parquet" -Epochs 1 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 1024 -Mode "audio_proj" -UseCompile 0
  Invoke-OmniTrain -LearningRate "5e-5" -DataPath "..\dataset\sft_a2a.parquet" -Epochs 3 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 1024 -Mode "all" -UseCompile 0
  Invoke-OmniTrain -LearningRate "5e-5" -DataPath "..\dataset\sft_i2t.parquet" -Epochs 1 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 768 -Mode "vision_proj" -UseCompile 1
  Invoke-OmniTrain -LearningRate "5e-6" -DataPath "..\dataset\sft_i2t.parquet" -Epochs 1 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 768 -Mode "all" -UseCompile 1
  Invoke-OmniTrain -LearningRate "5e-6" -DataPath "..\dataset\sft_a2a.parquet" -Epochs 1 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 1024 -Mode "all" -UseCompile 0
  Invoke-OmniTrain -LearningRate "5e-6" -DataPath "..\dataset\sft_i2t.parquet" -Epochs 1 -BatchSize 32 -FromWeight "sft_omni" -SaveWeight "sft_omni" -MaxSeqLen 768 -Mode "vision_proj" -UseCompile 1
}

function Convert-ScreenOmniTransformers {
  $outDir = Join-Path $miniMindORoot "out"
  $sourceWeight = Join-Path $outDir "zemo_screen_omni_768.pth"
  $finalWeight = Join-Path $outDir "zemo_screen_omni_final_768.pth"
  if (!(Test-Path $sourceWeight)) {
    throw "Screen MiniMind-O weight not found: $sourceWeight"
  }
  Copy-Item -LiteralPath $sourceWeight -Destination $finalWeight -Force

  $scriptDir = Join-Path $miniMindORoot "scripts"
  Push-Location $scriptDir
  try {
    Invoke-Py -c "from convert_omni import convert_torch2transformers; from model.model_omni import OmniConfig; convert_torch2transformers('../out/zemo_screen_omni_final_768.pth', '../zemo-screen-omni-final', OmniConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))"
  } finally {
    Pop-Location
  }
}

function Invoke-ScreenPipeline {
  param([string]$FromWeight)
  Export-ZeMoScreenData
  Invoke-OmniTrain -LearningRate "5e-5" -DataPath "..\dataset\zemo_screen_i2t.parquet" -Epochs $ScreenEpochs -BatchSize $ScreenBatchSize -FromWeight $FromWeight -SaveWeight "zemo_screen_omni" -MaxSeqLen 768 -Mode "vision_proj" -UseCompile 0
  Invoke-OmniTrain -LearningRate "5e-6" -DataPath "..\dataset\zemo_screen_i2t.parquet" -Epochs 1 -BatchSize $ScreenBatchSize -FromWeight "zemo_screen_omni" -SaveWeight "zemo_screen_omni" -MaxSeqLen 768 -Mode "all" -UseCompile 0
  Convert-ScreenOmniTransformers
}

Ensure-MiniMindO
Invoke-PythonInstall
Invoke-ModelDownload
Invoke-OfficialDataDownload
New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null

Push-Location $trainerDir
try {
  if ($Stage -eq "mini") {
    Invoke-MiniPipeline
  } elseif ($Stage -eq "full") {
    Invoke-FullPipeline
  } elseif ($Stage -eq "all") {
    Invoke-FullPipeline
    Invoke-ScreenPipeline -FromWeight "sft_omni"
  } else {
    Invoke-ScreenPipeline -FromWeight $ScreenFromWeight
  }
} finally {
  Pop-Location
}
