# ============================================================
# Trading Bot Arsenal - Unified Startup Script
# ============================================================
#
# Starts all trading bot components in visible windows:
# 1. Dashboard (Flask on port 5000/5001)
# 2. Telegram Bot (monitoring and alerts)
# 3. Master Orchestrator (live trading engine)
#
# Usage:
#   .\start.ps1                    # Start everything
#   .\start.ps1 -Capital 1000      # Custom capital
#   .\start.ps1 -Paper             # Paper mode (default)
#   .\start.ps1 -Live              # LIVE trading mode
#   .\start.ps1 -Status            # Check what's running
#   .\start.ps1 -Stop              # Stop everything
#
# Author: Trading Bot Arsenal
# Created: February 2026
# ============================================================

param(
    [switch]$Status,
    [switch]$Stop,
    [switch]$Paper,
    [switch]$Live,
    [int]$Capital = 500,
    [switch]$RotateLogs
)

$ErrorActionPreference = "Continue"
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

# ============================================================
# CONFIGURATION
# ============================================================

$Config = @{
    PythonExe = "python"
    DashboardScript = "dashboard\app.py"
    TelegramScript = "telegram_monitor.py"
    OrchestratorScript = "master_orchestrator.py"
    LogDir = "$ScriptDir\logs"
    PidFile = "$ScriptDir\logs\pids.json"
    LogArchiveDir = "$ScriptDir\logs\archive"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function Write-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Cyan
    Write-Host "     TRADING BOT ARSENAL" -ForegroundColor White
    Write-Host "     Unified Startup System" -ForegroundColor Gray
    Write-Host "  ============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success { param([string]$Message) Write-Host "  [OK] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "  [..] $Message" -ForegroundColor Cyan }
function Write-Warn { param([string]$Message) Write-Host "  [!] $Message" -ForegroundColor Yellow }
function Write-Err { param([string]$Message) Write-Host "  [X] $Message" -ForegroundColor Red }

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

# ============================================================
# PROCESS MANAGEMENT
# ============================================================

function Get-TradingBotProcesses {
    # Find all python processes related to trading_bot
    $processes = Get-Process -Name "python*" -ErrorAction SilentlyContinue | Where-Object {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
            $cmdLine -like "*trading_bot*" -or
            $cmdLine -like "*master_orchestrator*" -or
            $cmdLine -like "*telegram_monitor*" -or
            $cmdLine -like "*dashboard*app.py*"
        } catch {
            $false
        }
    }
    return $processes
}

function Stop-TradingBotProcesses {
    Write-Info "Stopping existing trading bot processes..."

    $processes = Get-TradingBotProcesses

    if ($processes.Count -eq 0) {
        Write-Success "No trading bot processes found"
        return
    }

    foreach ($proc in $processes) {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" -ErrorAction SilentlyContinue).CommandLine
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            Write-Success "Stopped PID $($proc.Id): $($cmdLine.Substring(0, [Math]::Min(60, $cmdLine.Length)))..."
        } catch {
            Write-Warn "Could not stop PID $($proc.Id): $_"
        }
    }

    Start-Sleep -Seconds 2
}

function Restart-DockerContainers {
    Write-Info "Restarting Docker containers..."

    # Check if Docker is running
    $dockerRunning = $false
    try {
        $dockerStatus = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerRunning = $true
        }
    } catch {
        Write-Warn "Docker not available"
        return
    }

    if (-not $dockerRunning) {
        Write-Warn "Docker daemon not running"
        return
    }

    # Restart kalshi_bot container
    try {
        $kalshiStatus = docker ps -a --filter "name=kalshi_bot" --format "{{.Status}}" 2>$null
        if ($kalshiStatus) {
            docker restart kalshi_bot 2>$null | Out-Null
            Write-Success "Restarted kalshi_bot container"
        } else {
            Write-Warn "kalshi_bot container not found"
        }
    } catch {
        Write-Warn "Failed to restart kalshi_bot: $_"
    }

    # Restart oanda_bot container
    try {
        $oandaStatus = docker ps -a --filter "name=oanda_bot" --format "{{.Status}}" 2>$null
        if ($oandaStatus) {
            docker restart oanda_bot 2>$null | Out-Null
            Write-Success "Restarted oanda_bot container"
        } else {
            Write-Warn "oanda_bot container not found"
        }
    } catch {
        Write-Warn "Failed to restart oanda_bot: $_"
    }
}

# ============================================================
# LOG MANAGEMENT
# ============================================================

function Rotate-Logs {
    Write-Info "Rotating log files..."

    Ensure-Directory $Config.LogArchiveDir

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFiles = Get-ChildItem -Path $Config.LogDir -Filter "*.log" -File -ErrorAction SilentlyContinue

    foreach ($log in $logFiles) {
        if ($log.Length -gt 10MB) {
            $archiveName = "$($log.BaseName)_$timestamp$($log.Extension)"
            $archivePath = Join-Path $Config.LogArchiveDir $archiveName
            Move-Item -Path $log.FullName -Destination $archivePath -Force
            Write-Success "Archived $($log.Name) ($('{0:N2}' -f ($log.Length/1MB)) MB)"
        }
    }

    # Clear old archive files (keep last 7 days)
    $cutoffDate = (Get-Date).AddDays(-7)
    Get-ChildItem -Path $Config.LogArchiveDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoffDate } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Clear-OldLogs {
    Write-Info "Clearing old log files..."

    $logFiles = @(
        "dashboard_out.log",
        "dashboard_err.log",
        "orchestrator_out.log",
        "orchestrator_err.log",
        "telegram_out.log",
        "telegram_err.log"
    )

    foreach ($logFile in $logFiles) {
        $logPath = Join-Path $Config.LogDir $logFile
        if (Test-Path $logPath) {
            Clear-Content $logPath -ErrorAction SilentlyContinue
        }
    }

    Write-Success "Log files cleared"
}

# ============================================================
# STARTUP FUNCTIONS
# ============================================================

function Start-Dashboard {
    Write-Info "Starting Dashboard..."

    $title = "Trading Dashboard"
    $script = Join-Path $ScriptDir $Config.DashboardScript

    $cmd = @"
`$Host.UI.RawUI.WindowTitle = '$title'
Write-Host '============================================' -ForegroundColor Cyan
Write-Host '  TRADING DASHBOARD' -ForegroundColor White
Write-Host '  Port: 5000 | http://localhost:5000' -ForegroundColor Gray
Write-Host '============================================' -ForegroundColor Cyan
Write-Host ''
Set-Location '$ScriptDir'
`$env:PYTHONIOENCODING = 'utf-8'
python '$script' 2>&1 | Out-Host
Write-Host ''
Write-Host 'Dashboard stopped. Press any key to close...' -ForegroundColor Yellow
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@

    $encodedCmd = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))

    $proc = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $encodedCmd -PassThru

    if ($proc) {
        Write-Success "Dashboard started (PID: $($proc.Id))"
        Write-Host "       URL: http://localhost:5000" -ForegroundColor Gray
        return $proc.Id
    } else {
        Write-Err "Failed to start Dashboard"
        return $null
    }
}

function Start-TelegramBot {
    Write-Info "Starting Telegram Bot..."

    $title = "Telegram Bot"
    $script = Join-Path $ScriptDir $Config.TelegramScript

    $cmd = @"
`$Host.UI.RawUI.WindowTitle = '$title'
Write-Host '============================================' -ForegroundColor Magenta
Write-Host '  TELEGRAM BOT MONITOR' -ForegroundColor White
Write-Host '  Commands: /status /help /positions' -ForegroundColor Gray
Write-Host '============================================' -ForegroundColor Magenta
Write-Host ''
Set-Location '$ScriptDir'
python '$script'
Write-Host ''
Write-Host 'Telegram bot stopped. Press any key to close...' -ForegroundColor Yellow
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@

    $encodedCmd = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))

    $proc = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $encodedCmd -PassThru

    if ($proc) {
        Write-Success "Telegram Bot started (PID: $($proc.Id))"
        return $proc.Id
    } else {
        Write-Err "Failed to start Telegram Bot"
        return $null
    }
}

function Start-Orchestrator {
    param(
        [int]$Capital = 500,
        [bool]$LiveMode = $false
    )

    Write-Info "Starting Master Orchestrator..."

    $title = "Master Orchestrator"
    $script = Join-Path $ScriptDir $Config.OrchestratorScript

    # Build arguments
    $modeArg = if ($LiveMode) { "--live --confirm-live" } else { "" }
    $args = "$modeArg --capital $Capital"

    $modeDisplay = if ($LiveMode) { "LIVE TRADING" } else { "PAPER MODE" }
    $modeColor = if ($LiveMode) { "Red" } else { "Green" }

    $cmd = @"
`$Host.UI.RawUI.WindowTitle = '$title'
Write-Host '============================================' -ForegroundColor $modeColor
Write-Host '  MASTER ORCHESTRATOR' -ForegroundColor White
Write-Host '  Mode: $modeDisplay | Capital: `$$Capital' -ForegroundColor Gray
Write-Host '============================================' -ForegroundColor $modeColor
Write-Host ''
Set-Location '$ScriptDir'
`$env:PYTHONIOENCODING = 'utf-8'
python '$script' $args 2>&1 | Out-Host
Write-Host ''
Write-Host 'Orchestrator stopped. Press any key to close...' -ForegroundColor Yellow
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@

    $encodedCmd = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))

    $proc = Start-Process powershell -ArgumentList "-NoExit", "-EncodedCommand", $encodedCmd -PassThru

    if ($proc) {
        $modeText = if ($LiveMode) { "LIVE" } else { "PAPER" }
        Write-Success "Orchestrator started (PID: $($proc.Id)) - $modeText mode, Capital: `$$Capital"
        return $proc.Id
    } else {
        Write-Err "Failed to start Orchestrator"
        return $null
    }
}

function Save-Pids {
    param($Pids)
    $Pids | ConvertTo-Json | Set-Content $Config.PidFile -Force
}

function Show-Status {
    Write-Banner
    Write-Host "  System Status" -ForegroundColor White
    Write-Host "  ============" -ForegroundColor Gray
    Write-Host ""

    # Check trading bot processes
    $processes = Get-TradingBotProcesses

    if ($processes.Count -eq 0) {
        Write-Host "  No trading bot processes running" -ForegroundColor Yellow
    } else {
        Write-Host "  Running Processes:" -ForegroundColor Cyan
        foreach ($proc in $processes) {
            try {
                $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" -ErrorAction SilentlyContinue).CommandLine
                $shortCmd = if ($cmdLine.Length -gt 70) { $cmdLine.Substring(0, 70) + "..." } else { $cmdLine }
                Write-Host "    PID $($proc.Id): $shortCmd" -ForegroundColor Green
            } catch {
                Write-Host "    PID $($proc.Id): (unknown)" -ForegroundColor Yellow
            }
        }
    }

    Write-Host ""

    # Check Docker containers
    Write-Host "  Docker Containers:" -ForegroundColor Cyan
    try {
        $containers = docker ps --filter "name=kalshi" --filter "name=oanda" --format "{{.Names}}: {{.Status}}" 2>$null
        if ($containers) {
            foreach ($container in $containers) {
                Write-Host "    $container" -ForegroundColor Green
            }
        } else {
            Write-Host "    No matching containers found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "    Docker not available" -ForegroundColor Yellow
    }

    Write-Host ""

    # Check ports
    Write-Host "  Network Ports:" -ForegroundColor Cyan
    $port5000 = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
    if ($port5000) {
        Write-Host "    Port 5000: IN USE (Dashboard)" -ForegroundColor Green
    } else {
        Write-Host "    Port 5000: Available" -ForegroundColor Yellow
    }

    Write-Host ""

    # Recent log activity
    Write-Host "  Recent Activity (master_orchestrator.log):" -ForegroundColor Cyan
    $logFile = Join-Path $Config.LogDir "master_orchestrator.log"
    if (Test-Path $logFile) {
        Get-Content $logFile -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "    $_" -ForegroundColor Gray
        }
    } else {
        Write-Host "    No log file found" -ForegroundColor Yellow
    }

    Write-Host ""
}

function Show-Summary {
    param(
        $DashboardPid,
        $OrchestratorPid,
        [bool]$LiveMode,
        [int]$Capital
    )

    $mode = if ($LiveMode) { "LIVE TRADING" } else { "PAPER MODE" }
    $modeColor = if ($LiveMode) { "Red" } else { "Green" }

    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor $modeColor
    Write-Host "     TRADING BOT ACTIVE - $mode" -ForegroundColor White
    Write-Host "  ============================================================" -ForegroundColor $modeColor
    Write-Host ""
    Write-Host "  Components Started:" -ForegroundColor Cyan

    if ($DashboardPid) {
        Write-Host "    [OK] Dashboard       PID: $DashboardPid   http://localhost:5000" -ForegroundColor Green
    } else {
        Write-Host "    [X] Dashboard        FAILED" -ForegroundColor Red
    }

    Write-Host "    [OK] Telegram App    (desktop app launched)" -ForegroundColor Green

    if ($OrchestratorPid) {
        Write-Host "    [OK] Orchestrator    PID: $OrchestratorPid   Capital: `$$Capital" -ForegroundColor Green
    } else {
        Write-Host "    [X] Orchestrator     FAILED" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "  Quick Commands:" -ForegroundColor Yellow
    Write-Host "    .\start.ps1 -Status     Check what's running" -ForegroundColor Gray
    Write-Host "    .\start.ps1 -Stop       Stop everything" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Logs: $($Config.LogDir)\" -ForegroundColor Gray
    Write-Host ""
}

# ============================================================
# MAIN EXECUTION
# ============================================================

# Handle -Stop
if ($Stop) {
    Write-Banner
    Stop-TradingBotProcesses
    Write-Host ""
    Write-Success "All trading bot processes stopped"
    Write-Host ""
    exit 0
}

# Handle -Status
if ($Status) {
    Show-Status
    exit 0
}

# Determine trading mode
$LiveMode = $Live -and -not $Paper
if ($Live -and $Paper) {
    Write-Warn "Both -Live and -Paper specified. Using Paper mode for safety."
    $LiveMode = $false
}

# Main startup sequence
Write-Banner

# Step 1: Stop existing processes
Stop-TradingBotProcesses

# Step 2: Restart Docker containers
Restart-DockerContainers

# Step 3: Rotate/clear logs
if ($RotateLogs) {
    Rotate-Logs
} else {
    Clear-OldLogs
}

# Step 4: Ensure directories exist
Ensure-Directory $Config.LogDir

Write-Host ""
Write-Host "  Starting Components..." -ForegroundColor White
Write-Host "  =====================" -ForegroundColor Gray
Write-Host ""

# Step 5: Start components with delays for proper initialization
$dashboardPid = Start-Dashboard
Start-Sleep -Seconds 3

# Open dashboard in Chrome
Write-Info "Opening dashboard in browser..."
try {
    Start-Process "chrome.exe" "http://localhost:5000" -ErrorAction SilentlyContinue
    Write-Success "Chrome opened with dashboard"
} catch {
    # Fallback to default browser
    Start-Process "http://localhost:5000"
    Write-Success "Browser opened with dashboard"
}

# Open Telegram desktop app (no bot monitor - just the app)
Write-Info "Opening Telegram desktop..."
$telegramPaths = @(
    "$env:APPDATA\Telegram Desktop\Telegram.exe",
    "$env:LOCALAPPDATA\Telegram Desktop\Telegram.exe",
    "C:\Program Files\Telegram Desktop\Telegram.exe",
    "$env:USERPROFILE\AppData\Roaming\Telegram Desktop\Telegram.exe"
)
$telegramOpened = $false
foreach ($tgPath in $telegramPaths) {
    if (Test-Path $tgPath) {
        Start-Process $tgPath -ErrorAction SilentlyContinue
        Write-Success "Telegram desktop opened"
        $telegramOpened = $true
        break
    }
}
if (-not $telegramOpened) {
    Write-Warn "Telegram desktop not found - check manually"
}

$orchestratorPid = Start-Orchestrator -Capital $Capital -LiveMode $LiveMode
Start-Sleep -Seconds 2

# Step 6: Save PIDs for future reference
$pids = @{
    dashboard = $dashboardPid
    orchestrator = $orchestratorPid
    startTime = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    mode = if ($LiveMode) { "live" } else { "paper" }
    capital = $Capital
}
Save-Pids $pids

# Step 7: Show summary
Show-Summary -DashboardPid $dashboardPid -OrchestratorPid $orchestratorPid -LiveMode $LiveMode -Capital $Capital

# Count failures
$failures = 0
if (-not $dashboardPid) { $failures++ }
if (-not $orchestratorPid) { $failures++ }

if ($failures -gt 0) {
    Write-Host "  [!] $failures component(s) failed to start. Check individual windows for errors." -ForegroundColor Yellow
    Write-Host ""
}

exit $failures
