# =============================================================================
# stop_cluster.ps1 — One-click Spark cluster shutdown across Windows lab machines
# =============================================================================
# Usage:
#   .\deploy\stop_cluster.ps1                # Stop all (workers first, then master)
#   .\deploy\stop_cluster.ps1 -WorkersOnly   # Stop workers, keep master running
#   .\deploy\stop_cluster.ps1 -SkipRemote    # Stop only local containers
#   .\deploy\stop_cluster.ps1 -Cleanup       # Stop + remove volumes and orphan containers
#
# Prerequisites:
#   - WinRM enabled on worker machines (for remote stop)
#     Enable with: Enable-PSRemoting -Force (run as Admin on each worker)
# =============================================================================

param(
    [switch]$WorkersOnly,
    [switch]$SkipRemote,
    [switch]$Cleanup
)

# =============================================================================
# CONFIGURATION — Must match start_cluster.ps1
# =============================================================================
$MASTER_IP = "192.168.1.100"

$WORKERS = @(
    @{ IP = $MASTER_IP;      Type = "cpu"; Location = "local";  Name = "spark-cpu-worker-local" },
    @{ IP = "192.168.1.101"; Type = "cpu"; Location = "remote"; Name = "spark-cpu-worker-1" },
    @{ IP = "192.168.1.102"; Type = "gpu"; Location = "remote"; Name = "spark-gpu-worker-1" }
)

# =============================================================================
# FUNCTIONS
# =============================================================================

function Write-Status($message, $color = "Cyan") {
    Write-Host "  [CLUSTER] $message" -ForegroundColor $color
}

function Stop-LocalContainer($name) {
    $exists = docker ps -a --filter "name=$name" --format "{{.Names}}" 2>$null
    if ($exists -eq $name) {
        Write-Status "Stopping: $name"
        docker stop $name 2>$null | Out-Null
        docker rm $name 2>$null | Out-Null
        Write-Status "$name stopped and removed" "Green"
        return $true
    } else {
        Write-Status "$name not found (already stopped)" "Yellow"
        return $false
    }
}

function Stop-RemoteContainer($workerIP, $name) {
    Write-Status "Stopping $name on $workerIP (remote)..."
    try {
        Invoke-Command -ComputerName $workerIP -ScriptBlock {
            param($containerName)
            docker stop $containerName 2>$null | Out-Null
            docker rm $containerName 2>$null | Out-Null
        } -ArgumentList $name -ErrorAction Stop
        Write-Status "$name stopped on $workerIP" "Green"
        return $true
    } catch {
        Write-Status "Failed to stop on ${workerIP}: $_" "Red"
        Write-Status "  Try manually: docker rm -f $name (on $workerIP)" "Yellow"
        return $false
    }
}

# =============================================================================
# MAIN
# =============================================================================

Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  SPARK INFERENCE CLUSTER — SHUTDOWN" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""

$stoppedCount = 0
$failedCount = 0

# --- Stop Workers First (graceful: workers deregister from master) ---
Write-Status "--- PHASE 1: STOPPING WORKERS ---"

foreach ($w in $WORKERS) {
    if ($w.Location -eq "remote" -and $SkipRemote) {
        Write-Status "Skipping remote: $($w.Name) ($($w.IP))" "Yellow"
        continue
    }

    if ($w.Location -eq "local") {
        $success = Stop-LocalContainer $w.Name
    } else {
        $success = Stop-RemoteContainer $w.IP $w.Name
    }

    if ($success) { $stoppedCount++ } else { $failedCount++ }
}

# Brief pause for workers to deregister
Start-Sleep -Seconds 2

# --- Stop Master ---
if (-not $WorkersOnly) {
    Write-Host ""
    Write-Status "--- PHASE 2: STOPPING MASTER ---"
    $success = Stop-LocalContainer "spark-master"
    if ($success) { $stoppedCount++ } else { $failedCount++ }
} else {
    Write-Host ""
    Write-Status "Workers-only mode. Master left running." "Yellow"
}

# --- Cleanup (optional) ---
if ($Cleanup) {
    Write-Host ""
    Write-Status "--- PHASE 3: CLEANUP ---"

    # Remove orphan containers from compose
    Write-Status "Removing orphan compose containers..."
    docker compose -f deploy/docker-compose.cluster.yml down --remove-orphans 2>$null | Out-Null

    # Remove dangling containers with spark- prefix
    $danglingContainers = docker ps -a --filter "name=spark-" --format "{{.Names}}" 2>$null
    if ($danglingContainers) {
        foreach ($c in $danglingContainers) {
            Write-Status "Removing dangling: $c" "Yellow"
            docker rm -f $c 2>$null | Out-Null
        }
    }

    # Prune unused networks
    Write-Status "Pruning unused Docker networks..."
    docker network prune -f 2>$null | Out-Null

    Write-Status "Cleanup complete" "Green"
}

# --- Summary ---
Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  CLUSTER SHUTDOWN COMPLETE" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""
Write-Status "Containers stopped: $stoppedCount"
if ($failedCount -gt 0) {
    Write-Status "Containers failed:  $failedCount" "Red"
}
if (-not $WorkersOnly) {
    Write-Status "Master UI is now offline"
} else {
    Write-Status "Master still running at: http://${MASTER_IP}:8080"
}
Write-Host ""
Write-Status "To restart: .\deploy\start_cluster.ps1"
Write-Host ""
