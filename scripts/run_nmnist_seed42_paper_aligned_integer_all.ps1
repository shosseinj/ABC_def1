param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe'
$runner = Join-Path $projectRoot 'scripts\run_nmnist_seed42_paper_aligned_integer_attacks.py'
$trainer = Join-Path $projectRoot 'scripts\train_nmnist_seed42_paper_aligned_integer.py'
$cleanResult = Join-Path $projectRoot 'Reports\results\nmnist_seed42_paper_aligned_integer\seed42_clean_result.json'
$stateDir = Join-Path $projectRoot 'Reports\checkpoints\nmnist_seed42_paper_aligned_integer'
$resultDir = Join-Path $projectRoot 'Reports\results\nmnist_seed42_paper_aligned_integer\attacks'
$logDir = Join-Path $projectRoot 'Reports\logs\nmnist_seed42_paper_aligned_integer'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Required Python interpreter not found: $python"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Attack runner not found: $runner"
}

function Assert-CompletedRun {
    param([string]$RunId)
    $markerPath = Join-Path $stateDir "$RunId.complete.json"
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) { return $false }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if ($marker.status -ne 'PASS' -or $marker.run_id -ne $RunId) {
        throw "Invalid completion marker: $markerPath"
    }
    $metadataPath = Join-Path $resultDir "$RunId.json"
    $artifactPath = Join-Path $resultDir "$RunId.npz"
    $auditPath = Join-Path $resultDir "$RunId.audit.json"
    foreach ($path in @($metadataPath, $artifactPath, $auditPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Completion marker exists but artifact is missing: $path"
        }
    }
    $checks = @(
        @($metadataPath, $marker.metadata_sha256),
        @($artifactPath, $marker.artifact_sha256),
        @($auditPath, $marker.audit_sha256)
    )
    foreach ($check in $checks) {
        $actual = (Get-FileHash -LiteralPath $check[0] -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$check[1]) {
            throw "Hash mismatch for completed artifact: $($check[0])"
        }
    }
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    if ($audit.status -ne 'PASS' -or $audit.run_id -ne $RunId -or
        $audit.clean_correct_denominator -ne 1000) {
        throw "Independent audit is not valid for $RunId"
    }
    return $true
}

$conditions = @(
    @{ Kind = 'B_inf'; Budget = 1 },
    @{ Kind = 'B_inf'; Budget = 2 },
    @{ Kind = 'B_inf'; Budget = 3 },
    @{ Kind = 'B1'; Budget = 500 },
    @{ Kind = 'B1'; Budget = 750 },
    @{ Kind = 'B1'; Budget = 1000 },
    @{ Kind = 'B1'; Budget = 1500 },
    @{ Kind = 'B0'; Budget = 200 },
    @{ Kind = 'B0'; Budget = 300 },
    @{ Kind = 'B0'; Budget = 400 },
    @{ Kind = 'B0'; Budget = 600 }
)

if (-not $DryRun) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $cleanResult -PathType Leaf)) {
        if ($DryRun) {
            Write-Host 'WOULD TRAIN seed-42 Integer count model and freeze clean-correct manifest'
        } else {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            $relativeLog = "Reports/logs/nmnist_seed42_paper_aligned_integer/train_${stamp}.log"
            $ErrorActionPreference = 'Continue'
            try {
                & $python -u $trainer --log $relativeLog 2>&1 | ForEach-Object { Write-Host $_ }
                $trainExit = $LASTEXITCODE
            } finally { $ErrorActionPreference = 'Stop' }
            if ($trainExit -ne 0) { throw "Integer training failed with exit code $trainExit" }
        }
    } elseif (-not $DryRun) {
        $clean = Get-Content -LiteralPath $cleanResult -Raw | ConvertFrom-Json
        if (-not $clean.clean_gate_pass) { throw 'Integer clean accuracy gate did not pass' }
        foreach ($entry in @(@($clean.checkpoint_path, $clean.checkpoint_sha256), @($clean.test_cache.path, $clean.test_cache.sha256), @($clean.predictions_path, $clean.predictions_sha256))) {
            $actual = (Get-FileHash -LiteralPath (Join-Path $projectRoot $entry[0]) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne [string]$entry[1]) { throw "Clean artifact hash mismatch: $($entry[0])" }
        }
        Write-Host 'SKIP clean training (checkpoint, cache and predictions hash verified)'
    }
    foreach ($condition in $conditions) {
        $kind = $condition.Kind
        $budget = $condition.Budget
        $runId = "seed42_paper_aligned_integer_${kind}_${budget}"
        if (Assert-CompletedRun -RunId $runId) {
            Write-Host "SKIP $runId (completion hashes and audit PASS)"
            continue
        }
        if ($DryRun) {
            Write-Host "WOULD RUN $runId"
            continue
        }
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $logPath = Join-Path $logDir "${runId}_${stamp}.log"
        Write-Host "START $runId | log: $logPath"
        $ErrorActionPreference = 'Continue'
        try {
            & $python -u $runner --budget-type $kind --budget $budget 2>&1 |
                Tee-Object -FilePath $logPath
            $attackExit = $LASTEXITCODE
        } finally { $ErrorActionPreference = 'Stop' }
        if ($attackExit -ne 0) {
            throw "$runId failed with exit code $attackExit; see $logPath"
        }
        if (-not (Assert-CompletedRun -RunId $runId)) {
            throw "$runId exited successfully without a validated completion marker"
        }
        Write-Host "PASS $runId"
    }
    if ($DryRun) {
        Write-Host 'Dry run complete; no experiment was executed.'
    } else {
        Write-Host 'All eleven N-MNIST seed-42 paper-aligned Integer conditions have valid completion markers.'
    }
} finally {
    Pop-Location
}
