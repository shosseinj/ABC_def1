param(
    [switch]$RunMissing,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe'
$updater = Join-Path $projectRoot 'scripts\update_nmnist_paper_tables.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Required interpreter missing: $python" }

Push-Location $projectRoot
try {
    if ($RunMissing) {
        if ($CheckOnly) { throw '-RunMissing and -CheckOnly cannot be combined' }
        $binaryClean = Join-Path $projectRoot 'Reports\results\nmnist_seed42_paper_aligned\seed42_clean_result.json'
        if (-not (Test-Path -LiteralPath $binaryClean -PathType Leaf)) {
            $trainer = Join-Path $projectRoot 'scripts\train_nmnist_seed42_paper_aligned.py'
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            $relativeLog = "Reports/logs/nmnist_seed42_paper_aligned/train_${stamp}.log"
            $ErrorActionPreference = 'Continue'
            try {
                & $python -u $trainer --log $relativeLog 2>&1 | ForEach-Object { Write-Host $_ }
                $trainExit = $LASTEXITCODE
            } finally { $ErrorActionPreference = 'Stop' }
            if ($trainExit -ne 0) { throw "Binary clean training failed with exit code $trainExit" }
        }
        & (Join-Path $projectRoot 'scripts\run_nmnist_seed42_paper_aligned_all.ps1')
        & (Join-Path $projectRoot 'scripts\run_nmnist_seed42_paper_aligned_integer_all.ps1')
    }
    if ($CheckOnly) {
        & $python $updater --check
    } else {
        & $python $updater
    }
    if ($LASTEXITCODE -ne 0) { throw "Table update failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
