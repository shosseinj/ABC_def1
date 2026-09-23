param([Parameter(Mandatory=$true)][int]$Phase)
$ErrorActionPreference = 'Stop'
if (Test-Path .\.venv\Scripts\Activate.ps1) { & .\.venv\Scripts\Activate.ps1 }
python phase_runner.py --phase $Phase
