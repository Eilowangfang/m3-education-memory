param(
    [string]$Model = "qwen3.7-plus",
    [int]$Limit = 1,
    [int]$MaxWorkers = 1,
    [switch]$StratifiedPilot
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$databasePath = Join-Path $projectRoot "data\education_memory.db"
$dependencyPath = Join-Path (Split-Path -Parent $projectRoot) "dataset\.python-libs"

$apiKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "DASHSCOPE_API_KEY is not configured. Run .\scripts\configure_bailian_key.ps1 first."
}
$env:DASHSCOPE_API_KEY = $apiKey
$baseUrl = if ($apiKey.StartsWith("sk-sp-")) {
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
} else {
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$fallbackPython = "C:\Users\wangf\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonPath = if (
    $pythonCommand -and
    $pythonCommand.Source -and
    $pythonCommand.Source -notlike "*\Microsoft\WindowsApps\python.exe"
) {
    $pythonCommand.Source
} else {
    $fallbackPython
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python was not found. Install Python 3.10+ or add python.exe to PATH."
}
if (-not (Test-Path -LiteralPath $databasePath)) {
    throw "Database not found: $databasePath"
}

$env:PYTHONPATH = "$projectRoot\vendor\python;$dependencyPath;$projectRoot\src"
Push-Location $projectRoot
try {
    $diagnoseArguments = @(
        "-m", "m3_edu_memory.cli", "diagnose",
        "--db", $databasePath,
        "--base-url", $baseUrl,
        "--model", $Model,
        "--api-key-env", "DASHSCOPE_API_KEY",
        "--limit", $Limit,
        "--max-workers", $MaxWorkers,
        "--only-unprocessed",
        "--audit-output-dir", (Join-Path $projectRoot "outputs\bailian-v2-audit"),
        "--result-output", (Join-Path $projectRoot "outputs\bailian-v2-audit\batch-results.json")
    )
    if ($StratifiedPilot) {
        $diagnoseArguments += "--stratified-pilot"
    }
    & $pythonPath @diagnoseArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
