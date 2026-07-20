# =============================================================================
# start_cluster.ps1 — One-click Spark cluster startup across Windows lab machines
# =============================================================================
# Usage:
#   .\deploy\start_cluster.ps1                    # Start with defaults
#   .\deploy\start_cluster.ps1 -MasterOnly        # Start master only
#   .\deploy\start_cluster.ps1 -SkipRemote        # Start master + local worker only
#
# Prerequisites:
#   - Docker Desktop running on all machines
#   - multi-model-inference:latest image loaded on all machines
#   - WinRM enabled on worker machines (for remote start)
#     Enable with: Enable-PSRemoting -Force (run as Admin on each worker)
#   - All machines on same subnet with firewall ports open
# =============================================================================

param(
    [switch]$MasterOnly,
    [switch]$SkipRemote,
    [switch]$Force
)

# =============================================================================
# CONFIGURATION — Edit these values for your lab
# =============================================================================
$MASTER_IP = "192.168.1.100"       # This machine's LAN IP
$IMAGE_NAME = "multi-model-inference:latest"

# Worker definitions: add/remove entries as your lab grows
# Type: "cpu" or "gpu"
# Cores: CPU cores to offer Spark
# Memory: RAM to offer Spark (use g suffix)
# Location: "local" (same machine as master) or "remote"
$WORKERS = @(
    @{ IP = $MASTER_IP;      Type = "cpu"; Cores = 2; Memory = "4g";  Location = "local";  Name = "spark-cpu-worker-local" },
    @{ IP = "192.168.1.101"; Type = "cpu"; Cores = 4; Memory = "8g";  Location = "remote"; Name = "spark-cpu-worker-1" },
    @{ IP = "192.168.1.102"; Type = "gpu"; Cores = 4; Memory = "12g"; Location = "remote"; Name = "spark-gpu-worker-1" }
)

# =============================================================================
# FUNCTIONS
# =============================================================================

function Write-Status($message, $color = "Cyan") {
    Write-Host "  [CLUSTER] $message" -ForegroundColor $color
}

function Test-ContainerRunning($name) {
    $result = docker ps --filter "name=$name" --format "{{.Names}}" 2>$null
    return ($result -eq $name)
}

function Remove-ExistingContainer($name) {
    $exists = docker ps -a --filter "name=$name" --format "{{.Names}}" 2>$null
    if ($exists -eq $name) {
        Write-Status "Removing existing container: $name" "Yellow"
        docker rm -f $name 2>$null | Out-Null
    }
}

function Wait-ForMaster($timeoutSec = 30) {
    Write-Status "Waiting for master to be ready..."
    for ($i = 0; $i -lt $timeoutSec; $i += 2) {
        Start-Sleep -Seconds 2
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8080/json/" -TimeoutSec 2 -ErrorAction Stop
            if ($response) {
                Write-Status "Master is ready!" "Green"
                return $true
            }
        } catch {
            # Not ready yet
        }
    }
    Write-Status "Master did not respond within ${timeoutSec}s" "Red"
    return $false
}

function Start-LocalContainer($name, $command, $extraArgs = "") {
    Remove-ExistingContainer $name
    $fullCmd = "docker run -d --name $name $extraArgs $IMAGE_NAME bash -c `"$command`""
    Write-Status "Starting: $name"
    Invoke-Expression $fullCmd | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "$name started successfully" "Green"
        return $true
    } else {
        Write-Status "Failed to start $name" "Red"
        return $false
    }
}

function Start-RemoteContainer($workerIP, $name, $command, $extraArgs = "") {
    $fullCmd = "docker rm -f $name 2>`$null; docker run -d --name $name $extraArgs $IMAGE_NAME bash -c `"$command`""
    Write-Status "Starting $name on $workerIP (remote)..."
    try {
        Invoke-Command -ComputerName $workerIP -ScriptBlock {
            param($cmd)
            Invoke-Expression $cmd
        } -ArgumentList $fullCmd -ErrorAction Stop | Out-Null
        Write-Status "$name started on $workerIP" "Green"
        return $true
    } catch {
        Write-Status "Failed to start on ${workerIP}: $_" "Red"
        Write-Status "  Hint: Run 'Enable-PSRemoting -Force' as Admin on $workerIP" "Yellow"
        return $false
    }
}

# =============================================================================
# MAIN
# =============================================================================

Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  SPARK INFERENCE CLUSTER — STARTUP" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""

# --- Validate Docker ---
$dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
if (-not $dockerVersion) {
    Write-Status "Docker is not running! Start Docker Desktop first." "Red"
    exit 1
}
Write-Status "Docker version: $dockerVersion"

# --- Validate image exists ---
$imageExists = docker images $IMAGE_NAME --format "{{.Repository}}" 2>$null
if (-not $imageExists) {
    Write-Status "Image '$IMAGE_NAME' not found. Build it first:" "Red"
    Write-Status "  docker build -t $IMAGE_NAME -f deploy/Dockerfile ." "Yellow"
    exit 1
}
Write-Status "Image found: $IMAGE_NAME"

# --- Start Master ---
Write-Host ""
Write-Status "--- PHASE 1: SPARK MASTER ---"

if ((Test-ContainerRunning "spark-master") -and -not $Force) {
    Write-Status "spark-master is already running (use -Force to restart)" "Yellow"
} else {
    $masterCmd = "start-master.sh -h $MASTER_IP && tail -f /opt/spark/logs/*master*"
    $masterArgs = "-p 7077:7077 -p 8080:8080 -p 4040:4040"
    Start-LocalContainer "spark-master" $masterCmd $masterArgs
}

# Wait for master
$masterReady = Wait-ForMaster
if (-not $masterReady) {
    Write-Status "Aborting — master not healthy" "Red"
    exit 1
}

if ($MasterOnly) {
    Write-Host ""
    Write-Status "Master-only mode. Workers skipped." "Yellow"
    Write-Status "Master UI: http://${MASTER_IP}:8080"
    exit 0
}

# --- Start Workers ---
Write-Host ""
Write-Status "--- PHASE 2: SPARK WORKERS ---"

$startedWorkers = 0
$failedWorkers = 0

foreach ($w in $WORKERS) {
    if ($w.Location -eq "remote" -and $SkipRemote) {
        Write-Status "Skipping remote worker: $($w.Name) ($($w.IP))" "Yellow"
        continue
    }

    # Build docker run arguments
    $gpuArgs = ""
    if ($w.Type -eq "gpu") {
        $gpuArgs = "--gpus all --shm-size=4g"
    }

    $portArgs = "-p 8081:8081"
    $envArgs = "-e SPARK_WORKER_HOST=$($w.IP)"
    $workerCmd = "SPARK_LOCAL_IP=$($w.IP) start-worker.sh spark://${MASTER_IP}:7077 -c $($w.Cores) -m $($w.Memory) && tail -f /opt/spark/logs/*worker*"
    $allArgs = "$portArgs $envArgs $gpuArgs"

    if ($w.Location -eq "local") {
        $success = Start-LocalContainer $w.Name $workerCmd $allArgs
    } else {
        $success = Start-RemoteContainer $w.IP $w.Name $workerCmd $allArgs
    }

    if ($success) { $startedWorkers++ } else { $failedWorkers++ }
}

# --- Summary ---
Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  CLUSTER STARTUP COMPLETE" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""
Write-Status "Master:          spark://${MASTER_IP}:7077"
Write-Status "Master UI:       http://${MASTER_IP}:8080"
Write-Status "Workers started: $startedWorkers"
if ($failedWorkers -gt 0) {
    Write-Status "Workers failed:  $failedWorkers" "Red"
}
Write-Host ""
Write-Status "Run benchmarks:"
Write-Status "  docker exec -it spark-master bash -c `"SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 5000`""
Write-Status "  docker exec -it spark-master bash -c `"SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --incremental`""
Write-Host ""
