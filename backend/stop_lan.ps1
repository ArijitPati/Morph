# Stops the Morph LAN backend reliably.
# Kills by PORT OWNER (not by remembered PID): on this machine
# `python -m uvicorn` appears as a husk parent + worker child, and only the
# worker holds :8010 — killing the wrong PID leaves the port held and the
# next start fails with EADDRINUSE. This kills the listener plus any stale
# husks with the same command line.
# Usage: powershell -ExecutionPolicy Bypass -File stop_lan.ps1

$killed = @()
Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        $killed += $_.OwningProcess
    }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "-m uvicorn app\.main:app" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $killed += $_.ProcessId
    }
if ($killed.Count -eq 0) { "nothing listening on :8010" } else { "killed: $($killed -join ', ')" }
