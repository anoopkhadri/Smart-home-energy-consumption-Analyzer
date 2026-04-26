Param(
    [int]$days = 90,
    [string]$model = "random_forest",
    [int]$horizon = 24,
    [switch]$tune
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

$TuneFlag = ""
if ($tune) { $TuneFlag = "--tune" }

Write-Host "Running end-to-end pipeline..." -ForegroundColor Cyan
python run_all.py --days $days --model $model --horizon $horizon $TuneFlag

Write-Host "Done. Models saved in ./models and plots in ./plots" -ForegroundColor Green
