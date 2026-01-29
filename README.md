# Perf/W SLO Monitoring for AI Workloads (local InfluxDB 3 + Grafana)

A lightweight runner + dashboard for **performance-per-watt** benchmarking of AI-style workloads on a Windows machine.
It runs a controlled workload, samples system + GPU telemetry during the run, writes results to **InfluxDB 3**, and visualizes them live in **Grafana**.

This repo includes:
- `perfw_app.py`: a simple GUI to load/edit config and run/stop sessions
- `perfw_core.py`: benchmark runner + telemetry collection + InfluxDB 3 line-protocol writer
- `dashboards/perf-w-ai-dashboard.json`: a working Grafana dashboard (importable)
- `config/config.example.json`: starter config you can load in the GUI

## Prerequisites

- **Windows 11**
- **Python 3.11+** (recommended: 3.11 or 3.12) with **Tkinter** included  
  > If you installed Python from python.org and checked “tcl/tk”, you’re good.
- **InfluxDB 3** running locally (default in this project: `http://localhost:8181`)
- **Grafana** running locally (default: `http://localhost:3000`)
- An InfluxDB 3 API token exported as an env var:
  - `INFLUXDB3_AUTH_TOKEN`

Optional (for GPU telemetry / AI workloads):
- NVIDIA GPU drivers installed (for NVML metrics)
- PyTorch installed (see below)

## Quick Start (PowerShell)

1) Clone the repo:
```powershell
git clone <YOUR_REPO_URL>
cd perfw-ai-runner
```

2) Create a virtual environment + install deps:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Install PyTorch
- CPU-only example:
  ```powershell
  pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  ```
- If you want CUDA builds, use the official selector on pytorch.org and paste the command.

4) Set your Influx token (current shell session):
```powershell
$env:INFLUXDB3_AUTH_TOKEN = "YOUR_TOKEN_HERE"
```
To set it permanently:
```powershell
setx INFLUXDB3_AUTH_TOKEN "YOUR_TOKEN_HERE"
```

5) Run the app:
```powershell
python .\src\perfw_app.py
```

6) In the GUI:
- Click **Load Config** and pick `config/config.example.json`
- Edit settings if needed (workload, duration, sample rate, etc.)
- Click **RUN**

## Grafana Dashboard Import

1) Open Grafana at `http://localhost:3000`
2) Configure an InfluxDB datasource that points to your local InfluxDB 3 instance
   - Keep the datasource consistent with what you already have working locally.
3) Import the dashboard:
   - Dashboards → **New** → **Import**
   - Upload: `dashboards/perf-w-ai-dashboard.json`
   - Select your Influx datasource when prompted

## Project Layout

```
perfw-ai-runner/
  src/
    perfw_app.py
    perfw_core.py
  dashboards/
    perf-w-ai-dashboard.json
  config/
    config.example.json
  scripts/
    run.ps1
  requirements.txt
  README.md
```

## Notes / Safety
- Do **not** commit tokens. This repo reads the token from `INFLUXDB3_AUTH_TOKEN`.
- Local output artifacts are written under `runs/` (ignored by git).

## License
MIT (add one if you want to open-source it).
