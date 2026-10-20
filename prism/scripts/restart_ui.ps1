$ErrorActionPreference = "Stop"

function Get-PrismConfig {
    $cfg = [ordered]@{
        port = 8080
        python = $env:PRISM_PYTHON
        # prism/scripts -> prism -> repo
        repo = if ($env:PRISM_REPO) { $env:PRISM_REPO } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
    }
    $settingsPath = Join-Path $env:APPDATA "Prism\prism-shell.json"
    if (Test-Path $settingsPath) {
        try {
            $s = Get-Content $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($s.port) { $cfg.port = [int]$s.port }
            if ($s.pythonPath) { $cfg.python = [string]$s.pythonPath }
            if ($s.repoRoot) { $cfg.repo = [string]$s.repoRoot }
        } catch {}
    }
    if (-not $cfg.python) {
        $hint = "E:\dev\env\miniconda3\envs\SR\python.exe"
        if (Test-Path $hint) { $cfg.python = $hint } else { $cfg.python = "python" }
    }
    return $cfg
}

function Test-PrismHealth([int]$port) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/v1/health" -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Test-PrismLab([int]$port) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/lab/" -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Stop-PrismControlPlane([int]$port) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "[Prism] Stopping stale control plane pid=$($_.OwningProcess) on :$port"
        cmd /c "taskkill /PID $($_.OwningProcess) /F" | Out-Null
    }
    Start-Sleep -Milliseconds 500
}

function Test-IsPrismDeskCommand([string]$cmd) {
    if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
    if ($cmd -match "play\.py") { return $false }
    if ($cmd -match " -m prism\.control(\s|$)") { return $false }
    if ($cmd -match " -m prism\s+(serve|agent|run|status|export|stop)(\s|$)") { return $false }
    if ($cmd -match "prism\\scripts\\accept_prism") { return $false }
    # Default `python -m prism` is desk; also explicit desk module.
    if ($cmd -match " -m prism(\s|$)") { return $true }
    if ($cmd -match " -m prism\.desk(\s|$)") { return $true }
    return $false
}

function Get-PrismDeskProcesses {
    # Avoid WQL OR filters (unreliable across hosts); filter names in PowerShell.
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python(w)?\.exe$' -and
            (Test-IsPrismDeskCommand ([string]$_.CommandLine))
        }
}

function Stop-PrismDesk {
    $desk = @(Get-PrismDeskProcesses)
    if ($desk.Count -eq 0) {
        Write-Host "[Prism] No desk window process found."
        return
    }
    foreach ($p in $desk) {
        Write-Host "[Prism] Closing UI pid=$($p.ProcessId)"
        cmd /c "taskkill /PID $($p.ProcessId) /F" | Out-Null
    }
    $deadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-PrismDeskProcesses).Count -eq 0) { return }
        Start-Sleep -Milliseconds 200
    }
    Write-Host "[Prism] Warning: some UI processes may still be exiting."
}

try {
    $cfg = Get-PrismConfig
    Write-Host "[Prism] Restart UI only (Lab window). Player will keep running."
    Write-Host "[Prism] python=$($cfg.python)"
    Write-Host "[Prism] repo=$($cfg.repo)"

    Stop-PrismDesk

    $planeUp = Test-PrismHealth $cfg.port
    if ($planeUp -and -not (Test-PrismLab $cfg.port)) {
        Write-Host "[Prism] Control plane health OK but /lab/ is broken; restarting control plane..."
        Stop-PrismControlPlane $cfg.port
        $planeUp = $false
    }

    $args = @("-m", "prism", "--bind", "127.0.0.1", "--port", "$($cfg.port)")
    if ($planeUp) {
        $args += "--no-server"
        Write-Host "[Prism] Control plane still up on :$($cfg.port); reopening window..."
    } else {
        Write-Host "[Prism] Control plane down; starting server + window..."
    }

    $proc = Start-Process -FilePath $cfg.python -ArgumentList $args -WorkingDirectory $cfg.repo -PassThru
    if (-not $proc) {
        throw "failed to start $($cfg.python) $($args -join ' ')"
    }
    Write-Host "[Prism] Started UI pid=$($proc.Id)"

    # Desk may spawn the control plane asynchronously; give it a moment, then verify.
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        throw "UI process exited immediately (code=$($proc.ExitCode)). Check PySide6 / imports from repo root."
    }
    if (-not $planeUp) {
        $ok = $false
        for ($i = 0; $i -lt 40; $i++) {
            if (Test-PrismHealth $cfg.port) { $ok = $true; break }
            if ($proc.HasExited) {
                throw "UI exited while waiting for control plane (code=$($proc.ExitCode))"
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ok) {
            Write-Host "[Prism] Warning: control plane not healthy yet on :$($cfg.port) (window may still be connecting)."
        } else {
            Write-Host "[Prism] Control plane ready on :$($cfg.port)"
        }
    }

    Write-Host "[Prism] UI relaunched. Player was not restarted."
    exit 0
} catch {
    Write-Host "[Prism] Error: $($_.Exception.Message)"
    exit 3
}
