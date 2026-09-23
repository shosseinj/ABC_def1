$ErrorActionPreference = 'Stop'
Write-Host 'Creating Python 3.11 virtual environment...'

pip install -r requirements.txt
python scripts\check_environment.py
Write-Host ''
Write-Host 'Environment ready. Next: python phase_runner.py --phase 1'
