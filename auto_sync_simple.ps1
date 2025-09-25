# Simple Auto-Sync Script for Trae-Streamlit-GitHub Integration
param(
    [string]$WatchPath = ".",
    [int]$DelaySeconds = 3
)

# Set up Git function
function git {
    & "C:\Program Files\Git\bin\git.exe" @args
}

Write-Host "Starting Auto-Sync Watcher..." -ForegroundColor Green
Write-Host "Watching: $((Get-Location).Path)" -ForegroundColor Cyan
Write-Host "Delay: $DelaySeconds seconds" -ForegroundColor Cyan

# Track last sync time
$lastSyncTime = Get-Date

# File extensions to watch
$watchExtensions = @("*.py", "*.toml", "*.txt", "*.md", "*.json")

# Files to exclude
$excludePatterns = @("*.log", "*.tmp", "*~", "*.swp", "*.pyc", "__pycache__", ".git", "*.pth")

# Function to check if file should be synced
function Should-SyncFile($filePath) {
    $fileName = Split-Path $filePath -Leaf
    
    # Check exclude patterns
    foreach ($pattern in $excludePatterns) {
        if ($fileName -like $pattern) {
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
        Write-Host "Skipping sync (too recent)" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Auto-syncing changes..." -ForegroundColor Green
    Write-Host "Changed file: $(Split-Path $changedFile -Leaf)" -ForegroundColor Cyan
    
    try {
        # Add all changes
        git add .
        
        # Check if there are changes to commit
        $status = git status --porcelain
        if (-not $status) {
            Write-Host "No changes to commit" -ForegroundColor Green
            return
        }
        
        # Commit changes
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $commitMessage = "Auto-sync: Updated $(Split-Path $changedFile -Leaf) at $timestamp"
        
        git commit -m $commitMessage
        
        # Push to GitHub
        git push origin main
        
        Write-Host "Auto-sync completed!" -ForegroundColor Green
        
        $script:lastSyncTime = $currentTime
        
    } catch {
        Write-Host "Auto-sync failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Create file system watcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $WatchPath
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Event handler
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    
    if (Should-SyncFile $path) {
        Write-Host "File ${changeType}: $(Split-Path $path -Leaf)" -ForegroundColor Yellow
        Start-Sleep -Milliseconds 500
        Invoke-AutoSync $path
    }
}

# Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action

Write-Host "Auto-sync watcher is active!" -ForegroundColor Green

try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "Auto-sync watcher stopped" -ForegroundColor Red
}