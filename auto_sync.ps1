# Auto-Sync Script for Trae-Streamlit-GitHub Integration
# This script watches for file changes and automatically syncs with Git and GitHub

param(
    [string]$WatchPath = ".",
    [int]$DelaySeconds = 5
)

# Set up Git function
function git {
    & "C:\Program Files\Git\bin\git.exe" @args
}

Write-Host "Starting Auto-Sync Watcher..." -ForegroundColor Green
Write-Host "Watching: $((Get-Location).Path)" -ForegroundColor Cyan
Write-Host "Delay: $DelaySeconds seconds" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

# File extensions to watch
$watchExtensions = @("*.py", "*.toml", "*.txt", "*.md", "*.json", "*.yml", "*.yaml")

# Files to exclude from auto-sync
$excludePatterns = @(
    "*.log", "*.tmp", "*~", "*.swp", "*.pyc", "__pycache__", 
    ".git", "node_modules", "*.pth", "test_images"
)

# Create file system watcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $WatchPath
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Track last sync time to avoid rapid syncs
$lastSyncTime = Get-Date

# Function to check if file should be synced
function Should-SyncFile($filePath) {
    $fileName = Split-Path $filePath -Leaf
    $relativePath = $filePath.Replace((Get-Location).Path, "").TrimStart('\')
    
    # Check exclude patterns
    foreach ($pattern in $excludePatterns) {
        if ($fileName -like $pattern -or $relativePath -like "*$pattern*") {
            return $false
        }
    }
    
    # Check include extensions
    foreach ($ext in $watchExtensions) {
        if ($fileName -like $ext) {
            return $true
        }
    }
    
    return $false
}

# Function to perform auto-sync
function Invoke-AutoSync($changedFile) {
    $currentTime = Get-Date
    $timeDiff = ($currentTime - $lastSyncTime).TotalSeconds
    
    if ($timeDiff -lt $DelaySeconds) {
        Write-Host "Skipping sync (too recent: $([math]::Round($timeDiff, 1))s ago)" -ForegroundColor Yellow
        return
    }
    
    Write-Host "`nAuto-syncing changes..." -ForegroundColor Green
    Write-Host "Changed file: $changedFile" -ForegroundColor Cyan
    
    try {
        # Add all changes
        Write-Host "📦 Staging changes..." -ForegroundColor Blue
        git add .
        
        # Check if there are changes to commit
        $status = git status --porcelain
        if (-not $status) {
            Write-Host "✅ No changes to commit" -ForegroundColor Green
            return
        }
        
        # Commit changes
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $commitMessage = "Auto-sync: Updated $(Split-Path $changedFile -Leaf) at $timestamp"
        
        Write-Host "💾 Committing changes..." -ForegroundColor Blue
        git commit -m $commitMessage
        
        # Push to GitHub
        Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Blue
        git push origin main
        
        Write-Host "✅ Auto-sync completed successfully!" -ForegroundColor Green
        Write-Host "🌐 Streamlit Cloud will redeploy automatically" -ForegroundColor Magenta
        
        $script:lastSyncTime = $currentTime
        
    } catch {
        Write-Host "❌ Auto-sync failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Event handlers
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    
    if (Should-SyncFile $path) {
        Write-Host "📁 File $changeType: $(Split-Path $path -Leaf)" -ForegroundColor Yellow
        
        # Small delay to ensure file operations are complete
        Start-Sleep -Milliseconds 500
        
        Invoke-AutoSync $path
    }
}

# Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Renamed" -Action $action

Write-Host "✅ Auto-sync watcher is now active!" -ForegroundColor Green
Write-Host "💡 Make changes to your files and they'll be automatically synced!" -ForegroundColor Cyan

try {
    # Keep the script running
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Clean up
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "`nAuto-sync watcher stopped" -ForegroundColor Red
}