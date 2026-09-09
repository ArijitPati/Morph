# Starts the Morph backend for LAN access (second laptop on the same network).
# Usage: powershell -ExecutionPolicy Bypass -File run_lan.ps1
# The server binds 0.0.0.0:8010. Logs go to backend/logs/lan-server.log.
# Stop it with: Get-NetTCPConnection -LocalPort 8010 -State Listen |
#   ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

$ErrorActionPreference = "Stop"
$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $backendDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "lan-server.log"
$errFile = Join-Path $logDir "lan-server.err.log"
$py = Join-Path (Split-Path -Parent $backendDir) "venv\Scripts\python.exe"

$proc = Start-Process -FilePath $py `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010" `
    -WorkingDirectory $backendDir `
    -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
    -WindowStyle Hidden -PassThru
"BACKEND PID: $($proc.Id) (logs: $logFile)"
