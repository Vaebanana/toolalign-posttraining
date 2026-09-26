param(
    [ValidateSet("All", "Inference", "Convert", "Evaluate")]
    [string]$Stage = "All",
    [string]$RunName = "dpo_v1",
    [string]$Categories = "live_simple,live_multiple,parallel,parallel_multiple,irrelevance",
    [string]$ModelPath = "outputs\models\qwen3_4b_sft_v1_merged",
    [AllowEmptyString()]
    [string]$AdapterPath = "outputs\train\qwen3_4b_qlora_dpo_v1",
    [string]$RegistryName = "",
    [string]$DisplayName = "ToolAlign Qwen3-4B DPO v1 (FC)",
    [string]$BfclDataDir = "",
    [string]$InferencePython = "D:\w_app\miniconda\Miniconda3\envs\pt\python.exe",
    [string]$BfclPython = "D:\w_app\miniconda\Miniconda3\envs\bfcl\python.exe",
    [int]$ProgressEvery = 10,
    [int]$MaxSamplesPerCategory = 0,
    [switch]$PartialEval,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

if ($RunName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunName may contain only letters, digits, dot, underscore, and hyphen."
}
if (-not (Test-Path -LiteralPath $InferencePython -PathType Leaf)) {
    throw "Inference Python not found: $InferencePython"
}
if (-not (Test-Path -LiteralPath $BfclPython -PathType Leaf)) {
    throw "BFCL Python not found: $BfclPython"
}
if ($ProgressEvery -lt 0) {
    throw "ProgressEvery must be non-negative."
}
if ($MaxSamplesPerCategory -lt 0) {
    throw "MaxSamplesPerCategory must be non-negative."
}
if ($MaxSamplesPerCategory -gt 0 -and -not $PartialEval) {
    throw "Use -PartialEval with -MaxSamplesPerCategory (smoke runs are incomplete)."
}

if (-not $RegistryName) {
    $RegistryName = "toolalign-qwen3-4b-$($RunName.Replace('_', '-'))-FC"
}
if ($RegistryName -match '[_/\\]') {
    throw "RegistryName cannot contain underscores or slashes because BFCL rewrites them."
}

if (-not $BfclDataDir) {
    $BfclDataDir = (& $BfclPython -c "from bfcl_eval.constants.eval_config import PROMPT_PATH; print(PROMPT_PATH)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $BfclDataDir) {
        throw "Could not discover BFCL package data with $BfclPython."
    }
}
if (-not (Test-Path -LiteralPath $BfclDataDir -PathType Container)) {
    throw "BFCL data directory not found: $BfclDataDir"
}

$RunRoot = Join-Path $ProjectRoot "outputs\bfcl\$RunName"
$RawDir = Join-Path $RunRoot "raw"
$ResultDir = Join-Path $RunRoot "result"
$ScoreDir = Join-Path $RunRoot "score"

# Model files are already cached locally; keep a long run independent of the Hub.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"

function Invoke-CheckedPython {
    param(
        [string]$Python,
        [string[]]$Arguments,
        [string]$Label
    )
    Write-Output "[$(Get-Date -Format s)] $Label"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if ($Stage -in @("All", "Inference")) {
    $InferenceArgs = @(
        "-m", "src.eval.bfcl_inference",
        "--bfcl-data-dir", $BfclDataDir,
        "--categories", $Categories,
        "--raw-dir", $RawDir,
        "--model-name-or-path", $ModelPath,
        "--adapter-name-or-path", $AdapterPath,
        "--progress-every", "$ProgressEvery"
    )
    if ($MaxSamplesPerCategory -gt 0) {
        $InferenceArgs += @("--max-samples-per-category", "$MaxSamplesPerCategory")
    }
    if ($Overwrite) {
        $InferenceArgs += "--overwrite"
    }
    Invoke-CheckedPython $InferencePython $InferenceArgs "BFCL inference start"
}

if ($Stage -in @("All", "Convert")) {
    $ConvertArgs = @(
        "-m", "src.eval.bfcl_convert",
        "--bfcl-data-dir", $BfclDataDir,
        "--categories", $Categories,
        "--raw-dir", $RawDir,
        "--result-dir", $ResultDir,
        "--registry-name", $RegistryName
    )
    if ($PartialEval) {
        $ConvertArgs += "--allow-partial"
    }
    Invoke-CheckedPython $InferencePython $ConvertArgs "BFCL result conversion start"
}

if ($Stage -in @("All", "Evaluate")) {
    $EvaluateArgs = @(
        "-m", "src.eval.bfcl_evaluate",
        "--categories", $Categories,
        "--result-dir", $ResultDir,
        "--score-dir", $ScoreDir,
        "--registry-name", $RegistryName,
        "--display-name", $DisplayName
    )
    if ($PartialEval) {
        $EvaluateArgs += "--partial-eval"
    }
    Invoke-CheckedPython $BfclPython $EvaluateArgs "Official BFCL evaluation start"
}

Write-Output "[$(Get-Date -Format s)] BFCL pipeline complete"
Write-Output "run_root: $RunRoot"
