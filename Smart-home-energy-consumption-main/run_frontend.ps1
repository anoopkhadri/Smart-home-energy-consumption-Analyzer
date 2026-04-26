Param(
    [int]$days = 90
)

$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
if (!(Test-Path .\.venv\Scripts\Activate.ps1)) {
  Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1

Write-Host "Installing requirements (including Streamlit)..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment to prevent interactive prompts and open in headless mode
$env:PYTHONPATH = (Get-Location).Path
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
$env:STREAMLIT_SERVER_HEADLESS = "true"

Write-Host "Launching Streamlit app at http://localhost:8501" -ForegroundColor Green
streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
