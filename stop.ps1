# ============================================================
# Trading Bot Arsenal - Stop Script
# ============================================================
#
# Stops all trading bot components cleanly
#
# Usage:
#   .\stop.ps1          # Stop all trading bot processes
#   .\stop.ps1 -Force   # Force kill all processes
#
# Author: Trading Bot Arsenal
# Created: February 2026
# ============================================================

param(
    [switch]$Force
)

$ErrorActionPreference = "Continue"
$ScriptDir = $PSScriptRoot

function Write-Banner {
    Write-Host ""
    Write-Host "  ============================================================" -ForegroundColor Red
    Write-Host "     TRADING BOT SHUTDOWN" -ForegroundColor White
    Write-Host "  ============================================================" -ForegroundColor Red
    Write-Host ""
}

function Write-Success { param([string]$Message) Write-Host "  [OK] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "  [..] $Message" -ForegroundColor Cyan }
function Write-Warn { param([string]$Message) Write-Host "  [!] $Message" -ForegroundColor Yellow }

function Get-TradingBotProcesses {
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

Write-Banner

Write-Info "Finding trading bot processes..."

$processes = Get-TradingBotProcesses

if ($processes.Count -eq 0) {
    Write-Host "  No trading bot processes found" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

Write-Host "  Found $($processes.Count) process(es):" -ForegroundColor Cyan
Write-Host ""

foreach ($proc in $processes) {
    try {
        $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" -ErrorAction SilentlyContinue).CommandLine
        $shortCmd = if ($cmdLine.Length -gt 60) { $cmdLine.Substring(0, 60) + "..." } else { $cmdLine }

        if ($Force) {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } else {
            Stop-Process -Id $proc.Id -ErrorAction Stop
        }

        Write-Success "Stopped PID $($proc.Id): $shortCmd"
    } catch {
        Write-Warn "Could not stop PID $($proc.Id): $_"
    }
}

# Also try to close PowerShell windows by title
$windowTitles = @("Trading Dashboard", "Telegram Bot", "Master Orchestrator")
foreach ($title in $windowTitles) {
    $psProcs = Get-Process -Name "powershell*" -ErrorAction SilentlyContinue | Where-Object {
        $_.MainWindowTitle -eq $title
    }
    foreach ($proc in $psProcs) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Success "Closed window: $title"
        } catch {}
    }
}

Write-Host ""
Write-Success "Shutdown complete"
Write-Host ""

# Clean up PID file
$pidFile = "$ScriptDir\logs\pids.json"
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
