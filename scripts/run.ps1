\
    # Run the Perf/W runner GUI from repo root
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"

    # Jump to repo root (parent of this script's folder)
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    Set-Location $RepoRoot

    if (!(Test-Path ".\.venv\Scripts\python.exe")) {
      Write-Host "No .venv found. Create it first:" -ForegroundColor Yellow
      Write-Host "  py -3.11 -m venv .venv"
      Write-Host "  .\.venv\Scripts\Activate.ps1"
      Write-Host "  pip install -r requirements.txt"
      exit 1
    }

    # Activate venv
    . .\.venv\Scripts\Activate.ps1

    # Ensure token is set (runner requires it to write to Influx)
    if (-not $env:INFLUXDB3_AUTH_TOKEN -or $env:INFLUXDB3_AUTH_TOKEN.Trim().Length -lt 10) {
      Write-Host "INFLUXDB3_AUTH_TOKEN is not set in this shell." -ForegroundColor Yellow
      Write-Host 'Set it with: $env:INFLUXDB3_AUTH_TOKEN="YOUR_TOKEN"' -ForegroundColor Yellow
    }

    python .\src\perfw_app.py
