param(
    [switch]$FullBudgets,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Required interpreter missing: $python" }

function Invoke-Python([string]$Script, [string[]]$Arguments) {
    $command = @('-u', (Join-Path $root $Script)) + $Arguments
    Write-Host "START $python $($command -join ' ')"
    $ErrorActionPreference = 'Continue'
    try {
        & $python @command 2>&1 | ForEach-Object { Write-Host $_ }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = 'Stop' }
    if ($code -ne 0) { throw "$Script failed with exit code $code" }
}

Push-Location $root
try {
    foreach ($representation in @('binary', 'integer')) {
        foreach ($model in @('convnet', 'resnet18', 'vggsnn')) {
            $args = @('--representation', $representation, '--model', $model)
            if ($DryRun) { Write-Host "WOULD TRAIN $representation $model" }
            else { Invoke-Python 'scripts/train_nmnist_controlled_four_models.py' $args }
        }
        if ($DryRun) { Write-Host "WOULD FREEZE joint clean-correct $representation manifest" }
        else { Invoke-Python 'scripts/build_nmnist_controlled_manifest.py' @('--representation', $representation) }
        if ($FullBudgets) {
            $conditions = @(
                @{Kind='B_inf'; Budget=1}, @{Kind='B_inf'; Budget=2}, @{Kind='B_inf'; Budget=3},
                @{Kind='B1'; Budget=500}, @{Kind='B1'; Budget=750}, @{Kind='B1'; Budget=1000},
                @{Kind='B0'; Budget=200}, @{Kind='B0'; Budget=300}, @{Kind='B0'; Budget=400}
            )
            if ($representation -eq 'integer') {
                $conditions += @{Kind='B1'; Budget=1500}, @{Kind='B0'; Budget=600}
            }
        } else {
            $conditions = @(@{Kind='B_inf'; Budget=1}, @{Kind='B0'; Budget=200})
        }
        foreach ($condition in $conditions) {
            foreach ($model in @('custom', 'convnet', 'resnet18', 'vggsnn')) {
                $args = @('--representation', $representation, '--model', $model,
                          '--budget-type', $condition.Kind, '--budget', [string]$condition.Budget)
                if ($DryRun) { Write-Host "WOULD ATTACK $representation $model $($condition.Kind)=$($condition.Budget)" }
                else { Invoke-Python 'scripts/run_nmnist_controlled_attack.py' $args }
            }
        }
    }
    if ($DryRun) { Write-Host 'Dry run complete; no experiment was executed.' }
    else { Write-Host 'Controlled four-model paired experiment complete; inspect Reports/results/nmnist_controlled_four_models.' }
} finally {
    Pop-Location
}
