# =============================================================================
# run_all_cpu_tests.ps1 - Complete CPU Test Matrix
# =============================================================================

$COMPOSE_FILE = "deploy/docker-compose.cluster.yml"
$MASTER = "spark-master"
$ENV_PREFIX = "SPARK_MASTER_URL=spark://spark-master:7077 CUDA_VISIBLE_DEVICES=''"

Write-Host ""
Write-Host "============================================================"
Write-Host "  COMPLETE CPU TEST MATRIX"
Write-Host "============================================================"
Write-Host ""

# =============================================================================
# PHASE 1: MODE COMPARISON
# =============================================================================
Write-Host "=== PHASE 1: MODE COMPARISON ==="

Write-Host "  1.1 All modes - Small load"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=modes_small python benchmark/run_benchmark.py --mode all --signal-samples 1000 --image-samples 20 --detection-samples 5 --batch-size 64 --partitions 4"

Write-Host "  1.2 All modes - Medium load"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=modes_medium python benchmark/run_benchmark.py --mode all --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 4"

Write-Host "  1.3 Distributed - Large load"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=modes_large python benchmark/run_benchmark.py --mode distributed --signal-samples 10000 --image-samples 100 --detection-samples 20 --batch-size 64 --partitions 8"

# =============================================================================
# PHASE 2: PARTITION SCALING
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 2: PARTITION SCALING ==="

$partitionList = @(2, 4, 6, 8, 12, 16)
foreach ($p in $partitionList) {
    Write-Host "  2.$p Partitions=$p"
    docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=partitions_$p python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions $p"
    Start-Sleep -Seconds 3
}

# =============================================================================
# PHASE 3: DATA SIZE SCALING
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 3: DATA SIZE SCALING ==="

Write-Host "  3.1 Tiny"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=datasize_tiny python benchmark/run_benchmark.py --mode distributed --signal-samples 500 --image-samples 10 --detection-samples 5 --batch-size 64 --partitions 4"
Start-Sleep -Seconds 3

Write-Host "  3.2 Small"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=datasize_small python benchmark/run_benchmark.py --mode distributed --signal-samples 1000 --image-samples 20 --detection-samples 5 --batch-size 64 --partitions 4"
Start-Sleep -Seconds 3

Write-Host "  3.3 Medium"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=datasize_medium python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 4"
Start-Sleep -Seconds 3

Write-Host "  3.4 Large"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=datasize_large python benchmark/run_benchmark.py --mode distributed --signal-samples 8000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 8"
Start-Sleep -Seconds 3

Write-Host "  3.5 XLarge"
docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=datasize_xlarge python benchmark/run_benchmark.py --mode distributed --signal-samples 10000 --image-samples 80 --detection-samples 15 --batch-size 64 --partitions 8"
Start-Sleep -Seconds 3

# =============================================================================
# PHASE 4: BATCH SIZE IMPACT
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 4: BATCH SIZE IMPACT ==="

$batchList = @(16, 32, 64, 128, 256, 512)
foreach ($bs in $batchList) {
    Write-Host "  4. Batch size=$bs"
    docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=batch_$bs python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size $bs --partitions 4"
    Start-Sleep -Seconds 3
}

# =============================================================================
# PHASE 5: WORKER SCALING
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 5: WORKER SCALING ==="

$workerList = @(1, 2, 3, 4, 6)
foreach ($w in $workerList) {
    Write-Host "  5. Workers=$w - Restarting cluster..."
    docker compose -f $COMPOSE_FILE up -d --scale spark-cpu-worker=$w --scale spark-gpu-worker=0 2>$null
    Start-Sleep -Seconds 15

    $partForWorkers = $w * 2
    docker exec -it $MASTER bash -c "$ENV_PREFIX RUN_NAME=workers_$w python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions $partForWorkers"
    Start-Sleep -Seconds 5
}

# Restore default
Write-Host "  Restoring default workers..."
docker compose -f $COMPOSE_FILE up -d --scale spark-cpu-worker=2 --scale spark-gpu-worker=1 2>$null
Start-Sleep -Seconds 10

# =============================================================================
# PHASE 6: CLUSTER BENCHMARK - DEVICE MODES
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 6: CLUSTER BENCHMARK ==="

Write-Host "  6.1 cpu_only - 2 partitions"
docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 2 --signal-samples 3000"
Start-Sleep -Seconds 3

Write-Host "  6.2 cpu_only - 4 partitions"
docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 3000"
Start-Sleep -Seconds 3

Write-Host "  6.3 cpu_only - 8 partitions"
docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 8 --signal-samples 3000"
Start-Sleep -Seconds 3

Write-Host "  6.4 hybrid - 4 partitions"
docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 3000"
Start-Sleep -Seconds 3

# =============================================================================
# PHASE 7: INCREMENTAL LOAD TEST
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 7: INCREMENTAL LOAD TEST ==="

docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/incremental_load_test.py"
Start-Sleep -Seconds 5

# =============================================================================
# PHASE 8: FULL INCREMENTAL - ALL MODES x 3 LOADS
# =============================================================================
Write-Host ""
Write-Host "=== PHASE 8: FULL INCREMENTAL ==="

docker exec -it $MASTER bash -c "$ENV_PREFIX python benchmark/cluster_benchmark.py --incremental"

# =============================================================================
# DONE
# =============================================================================
Write-Host ""
Write-Host "============================================================"
Write-Host "  ALL TESTS COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "  Results in: results\"
Write-Host ""
