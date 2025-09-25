# Development Environment Startup Script
# Launches Streamlit app and auto-sync watcher for seamless development

Write-Host "🚀 Starting Leuko App Development Environment..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Set up Git function
function git {
    & "C:\Program Files\Git\bin\git.exe" @args
}

# Check if Streamlit is already running
$streamlitProcess = Get-Process | Where-Object { $_.ProcessName -eq "streamlit" -or $_.CommandLine -like "*streamlit*" }
if ($streamlitProcess) {
    Write-Host "⚠️  Streamlit is already running. Stopping existing processes..." -ForegroundColor Yellow
    $streamlitProcess | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Write-Host "📋 Environment Setup:" -ForegroundColor Cyan
Write-Host "   📁 Project: Leuko App Development" -ForegroundColor White
Write-Host "   🌐 Local URL: http://localhost:8504" -ForegroundColor White
Write-Host "   ☁️  Cloud URL: https://leukoappsaplc.streamlit.app" -ForegroundColor White
Write-Host "   📦 Repository: https://github.com/shawredanalytics/LeukoApp" -ForegroundColor White

Write-Host "`n🔧 Starting Services..." -ForegroundColor Green

# Start Streamlit in background
Write-Host "🌟 Starting Streamlit app..." -ForegroundColor Blue
$streamlitJob = Start-Job -ScriptBlock {
    Set-Location "c:\Users\MANIKUMAR\Desktop\Leuko App Dev"
    streamlit run app_binary_screening.py --server.port 8504
}

# Wait a moment for Streamlit to start
Start-Sleep -Seconds 3

# Check if Streamlit started successfully
$streamlitUrl = "http://localhost:8504"
try {
    $response = Invoke-WebRequest -Uri $streamlitUrl -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ Streamlit app is running at $streamlitUrl" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Streamlit may still be starting up..." -ForegroundColor Yellow
}

# Start auto-sync watcher
Write-Host "🔄 Starting auto-sync watcher..." -ForegroundColor Blue
$autoSyncJob = Start-Job -ScriptBlock {
    Set-Location "c:\Users\MANIKUMAR\Desktop\Leuko App Dev"
    .\auto_sync.ps1 -DelaySeconds 3
}

Start-Sleep -Seconds 2

Write-Host "`n✅ Development Environment Ready!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host "`n📊 Active Services:" -ForegroundColor Cyan
Write-Host "   🌟 Streamlit App: Job ID $($streamlitJob.Id)" -ForegroundColor White
Write-Host "   🔄 Auto-Sync: Job ID $($autoSyncJob.Id)" -ForegroundColor White

Write-Host "`n💡 Development Workflow:" -ForegroundColor Cyan
Write-Host "   1. Edit files in Trae IDE" -ForegroundColor White
Write-Host "   2. Save changes (Ctrl+S)" -ForegroundColor White
Write-Host "   3. Streamlit auto-reloads locally" -ForegroundColor White
Write-Host "   4. Changes auto-sync to GitHub" -ForegroundColor White
Write-Host "   5. Streamlit Cloud auto-deploys" -ForegroundColor White

Write-Host "`n🔗 Quick Links:" -ForegroundColor Cyan
Write-Host "   🖥️  Local App: $streamlitUrl" -ForegroundColor White
Write-Host "   ☁️  Live App: https://leukoappsaplc.streamlit.app" -ForegroundColor White
Write-Host "   📦 GitHub: https://github.com/shawredanalytics/LeukoApp" -ForegroundColor White

Write-Host "`n⚡ Commands:" -ForegroundColor Cyan
Write-Host "   📊 Check status: Get-Job" -ForegroundColor White
Write-Host "   🛑 Stop services: Stop-Job -Id <JobId>" -ForegroundColor White
Write-Host "   📋 View logs: Receive-Job -Id <JobId>" -ForegroundColor White
Write-Host "   🔄 Restart: .\start_dev_environment.ps1" -ForegroundColor White

Write-Host "`n🎯 Ready for development! Make changes and watch the magic happen!" -ForegroundColor Green

# Keep monitoring jobs
Write-Host "`nPress Ctrl+C to stop all services and exit..." -ForegroundColor Yellow

try {
    while ($true) {
        # Check job status every 30 seconds
        Start-Sleep -Seconds 30
        
        $jobs = Get-Job
        $runningJobs = $jobs | Where-Object { $_.State -eq "Running" }
        
        if ($runningJobs.Count -eq 0) {
            Write-Host "⚠️  All services stopped. Restarting..." -ForegroundColor Yellow
            break
        }
    }
} catch {
    Write-Host "`n🛑 Stopping development environment..." -ForegroundColor Red
} finally {
    # Clean up jobs
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    Write-Host "✅ Development environment stopped" -ForegroundColor Green
}