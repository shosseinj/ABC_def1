param(
    [ValidateSet('FairnessAudit', 'NmnistIntegerLocal', 'CleanTrainingLocal', 'AllLocal')]
    [string]$Stage = 'FairnessAudit',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe'
$logDir = Join-Path $projectRoot 'Reports\logs\manual_research_runner'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Required Python interpreter not found: $python"
}

function Invoke-Stage {
    param([string]$Name, [string]$Script, [string[]]$Arguments)
    $scriptPath = Join-Path $projectRoot $Script
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "Stage script not found: $scriptPath"
    }
    $display = "$python -u $scriptPath $($Arguments -join ' ')"
    if ($DryRun) {
        Write-Host "WOULD RUN $Name | $display"
        return
    }
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $logPath = Join-Path $logDir "${Name}_${stamp}.log"
    Write-Host "START $Name | $display"
    Write-Host "LOG $logPath"
    # Windows PowerShell wraps native stderr as an ErrorRecord. Numba warnings
    # must be logged, while the native process exit code remains authoritative.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $python -u $scriptPath @Arguments 2>&1 | Tee-Object -FilePath $logPath
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "$Name failed with exit code $nativeExitCode; see $logPath"
    }
    Write-Host "DONE $Name"
}

Push-Location $projectRoot
try {
    if ($Stage -eq 'FairnessAudit') {
        Write-Host 'N-MNIST Binary published-reference comparison is available in Reports\nmnist_published_reference_comparison.md.'
        Write-Host 'Local Integer attacks use equal-duration bins and do not match paper Table 2.'
    }
    if ($Stage -in @('NmnistIntegerLocal', 'AllLocal')) {
        Invoke-Stage -Name 'nmnist_integer_corrected' `
            -Script 'scripts\run_nmnist_integer_corrected.py' `
            -Arguments @('--all')
    }
    if ($Stage -in @('CleanTrainingLocal', 'AllLocal')) {
        Invoke-Stage -Name 'clean_improved_final' `
            -Script 'scripts\run_clean_improved_final.py' `
            -Arguments @()
    }
    if (-not $DryRun -and $Stage -ne 'FairnessAudit') {
        Write-Host 'Local stages finished. Their results are not an exact four-model paper comparison.'
    }
} finally {
    Pop-Location
}
