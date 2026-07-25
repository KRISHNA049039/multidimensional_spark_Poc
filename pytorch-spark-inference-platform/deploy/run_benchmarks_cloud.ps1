# =============================================================================
# run_benchmarks_cloud.ps1 — Spin up cost-effective EC2 spot instances,
# run ALL benchmark tests, download results, terminate instances.
#
# Cost strategy:
#   - Spot instances (60-90% discount vs on-demand)
#   - g4dn.2xlarge: 8 vCPUs, 32GB RAM, 1x T4 GPU — ~$0.23/hr spot
#   - Auto-terminate after tests complete
#   - Total estimated cost: $1-3 for full benchmark suite (~1-2 hours)
#
# Usage:
#   .\deploy\run_benchmarks_cloud.ps1
#   .\deploy\run_benchmarks_cloud.ps1 -Region us-east-1 -InstanceType g4dn.4xlarge
#   .\deploy\run_benchmarks_cloud.ps1 -SkipDeploy   # Re-run on existing instances
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate permissions
#   - Existing CDK stack deployed (or use -DeployStack flag)
# =============================================================================

param(
    [string]$Region = "us-east-1",
    [string]$MasterInstanceType = "c5.2xlarge",     # 8 vCPUs, 16GB — ~$0.10/hr spot
    [string]$WorkerInstanceType = "g4dn.2xlarge",   # 8 vCPUs, 32GB, 1x T4 GPU — ~$0.23/hr spot
    [int]$CpuWorkerCount = 1,                        # Additional CPU workers
    [string]$ResultsLocalPath = ".\results\cloud_benchmark",
    [switch]$SkipDeploy,
    [switch]$KeepRunning,                            # Don't terminate after tests
    [switch]$UseOnDemand                             # Use on-demand instead of spot
)

$ErrorActionPreference = "Stop"
$AWS = "aws"
$STACK_NAME = "SparkBenchmarkSpotStack"
$PROJECT_DIR = Split-Path -Parent $PSScriptRoot

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Step($phase, $message, $color = "Cyan") {
    Write-Host "  [$phase] $message" -ForegroundColor $color
}

function Wait-SSMCommand($commandId, $instanceId, $timeoutMin = 30) {
    $elapsed = 0
    $interval = 15
    while ($elapsed -lt ($timeoutMin * 60)) {
        Start-Sleep -Seconds $interval
        $elapsed += $interval
        $result = & $AWS ssm get-command-invocation `
            --command-id $commandId `
            --instance-id $instanceId `
            --region $Region `
            --output json 2>&1 | ConvertFrom-Json

        switch ($result.Status) {
            "Success" {
                Write-Step "SSM" "Command completed successfully" "Green"
                return $result
            }
            "Failed" {
                Write-Step "SSM" "Command FAILED" "Red"
                Write-Host $result.StandardErrorContent
                return $result
            }
            "TimedOut" {
                Write-Step "SSM" "Command timed out" "Red"
                return $result
            }
            default {
                $min = [math]::Floor($elapsed / 60)
                Write-Step "SSM" "Still running... (${min}m elapsed)" "Yellow"
            }
        }
    }
    Write-Step "SSM" "Local timeout after ${timeoutMin}m" "Red"
    return $null
}

function Get-StackOutput($outputKey) {
    $outputs = & $AWS cloudformation describe-stacks `
        --stack-name $STACK_NAME `
        --region $Region `
        --query "Stacks[0].Outputs[?OutputKey=='$outputKey'].OutputValue" `
        --output text 2>&1
    return $outputs
}

# =============================================================================
# PHASE 0: VALIDATE PREREQUISITES
# =============================================================================

Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  CLOUD BENCHMARK RUNNER — SPOT INSTANCES" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""
Write-Host "  Config:" -ForegroundColor Gray
Write-Host "    Region:          $Region" -ForegroundColor Gray
Write-Host "    Master:          $MasterInstanceType (spot)" -ForegroundColor Gray
Write-Host "    GPU Worker:      $WorkerInstanceType (spot)" -ForegroundColor Gray
Write-Host "    CPU Workers:     $CpuWorkerCount" -ForegroundColor Gray
Write-Host "    Estimated cost:  ~`$1-3 total" -ForegroundColor Gray
Write-Host ""

# Validate AWS CLI
try {
    $identity = & $AWS sts get-caller-identity --output json 2>&1 | ConvertFrom-Json
    Write-Step "INIT" "AWS Account: $($identity.Account)"
} catch {
    Write-Step "INIT" "AWS CLI not configured. Run 'aws configure' first." "Red"
    exit 1
}

# =============================================================================
# PHASE 1: DEPLOY INFRASTRUCTURE (CDK or CloudFormation)
# =============================================================================

if (-not $SkipDeploy) {
    Write-Host ""
    Write-Step "DEPLOY" "--- Deploying spot instance infrastructure ---"

    # Upload project to S3 first
    Write-Step "DEPLOY" "Creating project zip..."
    Set-Location $PROJECT_DIR
    if (Test-Path "project.zip") { Remove-Item "project.zip" -Force }

    python -c @"
import zipfile, os
exclude = {'.git', '.venv', 'cdk.out', 'node_modules', '__pycache__', 'project.zip', 'results'}
with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if f == 'project.zip':
                continue
            filepath = os.path.join(root, f)
            arcname = filepath.replace('\\', '/')
            if arcname.startswith('./'):
                arcname = arcname[2:]
            zf.write(filepath, arcname)
    print(f'  Zipped {len(zf.namelist())} files')
"@

    # Deploy the spot stack using CloudFormation directly (no CDK dependency needed)
    Write-Step "DEPLOY" "Deploying CloudFormation stack..."

    $spotPrice = if ($UseOnDemand) { "no" } else { "yes" }

    $templatePath = Join-Path $PROJECT_DIR "deploy\spot_benchmark_cfn.yaml"
    & $AWS cloudformation deploy `
        --template-file $templatePath `
        --stack-name $STACK_NAME `
        --region $Region `
        --capabilities CAPABILITY_IAM `
        --parameter-overrides `
            MasterInstanceType=$MasterInstanceType `
            WorkerInstanceType=$WorkerInstanceType `
            UseSpotInstances=$spotPrice `
        --no-fail-on-empty-changeset

    if ($LASTEXITCODE -ne 0) {
        Write-Step "DEPLOY" "CloudFormation deploy failed!" "Red"
        exit 1
    }

    # Get outputs
    $bucketName = Get-StackOutput "ArtifactsBucketName"
    Write-Step "DEPLOY" "Artifacts bucket: $bucketName"

    # Upload project zip
    Write-Step "DEPLOY" "Uploading project to S3..."
    & $AWS s3 cp project.zip "s3://$bucketName/inference/project.zip" --region $Region
    Remove-Item "project.zip" -Force

    # Wait for instances to be ready (SSM online)
    Write-Step "DEPLOY" "Waiting for instances to come online..."
    $masterInstanceId = Get-StackOutput "MasterInstanceId"
    $gpuWorkerInstanceId = Get-StackOutput "GpuWorkerInstanceId"

    Write-Step "DEPLOY" "Master:     $masterInstanceId"
    Write-Step "DEPLOY" "GPU Worker: $gpuWorkerInstanceId"

    # Wait for SSM connectivity
    $retries = 0
    $maxRetries = 40  # 10 minutes max
    do {
        Start-Sleep -Seconds 15
        $retries++
        $ssmStatus = & $AWS ssm describe-instance-information `
            --filters "Key=InstanceIds,Values=$masterInstanceId" `
            --region $Region --output json 2>&1 | ConvertFrom-Json
        $online = $ssmStatus.InstanceInformationList.Count -gt 0
        if (-not $online) {
            Write-Step "DEPLOY" "Waiting for SSM... ($($retries * 15)s)" "Yellow"
        }
    } while (-not $online -and $retries -lt $maxRetries)

    if (-not $online) {
        Write-Step "DEPLOY" "Instances not reachable via SSM after 10 min" "Red"
        exit 1
    }
    Write-Step "DEPLOY" "Instances online and ready" "Green"

} else {
    # SkipDeploy - get existing stack info
    $bucketName = Get-StackOutput "ArtifactsBucketName"
    $masterInstanceId = Get-StackOutput "MasterInstanceId"
    $gpuWorkerInstanceId = Get-StackOutput "GpuWorkerInstanceId"
    Write-Step "DEPLOY" "Using existing stack — Master: $masterInstanceId"

    # Always re-upload project zip so instances get latest code
    Write-Step "DEPLOY" "Re-uploading project zip with latest changes..."
    Set-Location $PROJECT_DIR
    if (Test-Path "project.zip") { Remove-Item "project.zip" -Force }
    python -c @"
import zipfile, os
exclude = {'.git', '.venv', 'cdk.out', 'node_modules', '__pycache__', 'project.zip', 'results'}
with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if f == 'project.zip': continue
            filepath = os.path.join(root, f)
            arcname = filepath.replace('\\', '/')
            if arcname.startswith('./'): arcname = arcname[2:]
            zf.write(filepath, arcname)
    print(f'  Zipped {len(zf.namelist())} files')
"@
    & $AWS s3 cp project.zip "s3://$bucketName/inference/project.zip" --region $Region
    Remove-Item "project.zip" -Force
    Write-Step "DEPLOY" "Project zip uploaded" "Green"
}

# =============================================================================
# PHASE 2: SETUP AND BUILD ON INSTANCES
# =============================================================================

Write-Host ""
Write-Step "SETUP" "--- Building Docker image on instances ---"

# Write setup scripts to temp files and upload to S3
$masterSetupContent = @"
#!/bin/bash
set -e
echo '=== Setting up Spark Benchmark ==='
export ARTIFACTS_BUCKET=$bucketName
export AWS_DEFAULT_REGION=$Region

mkdir -p /opt/spark-inference/app
aws s3 cp s3://$bucketName/inference/project.zip /opt/spark-inference/project.zip
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app

echo '=== Building Docker image ==='
docker build --no-cache --network host -t multi-model-inference:latest -f deploy/Dockerfile .

echo '=== Starting Spark Master ==='
docker rm -f spark-master spark-cpu-worker 2>/dev/null || true
MASTER_IP=`$(hostname -I | awk '{print `$1}')

docker run -d --name spark-master --network host -v /opt/spark-inference/app/results:/app/results multi-model-inference:latest bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"

sleep 10

docker run -d --name spark-cpu-worker --network host -v /opt/spark-inference/app/results:/app/results multi-model-inference:latest bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"

sleep 5
echo "MASTER_IP=`${MASTER_IP}"
echo '=== Master setup complete ==='
"@

$masterSetupPath = "$env:TEMP\setup_master_bench.sh"
[System.IO.File]::WriteAllText($masterSetupPath, $masterSetupContent.Replace("`r`n", "`n"))

Write-Step "SETUP" "Uploading setup script to S3..."
& $AWS s3 cp $masterSetupPath "s3://$bucketName/scripts/setup_master_bench.sh" --region $Region

# Send SSM command to download and run the script
Write-Step "SETUP" "Setting up master instance (building image, ~5-8 min)..."
$ssmParams = @{
    commands = @("aws s3 cp s3://$bucketName/scripts/setup_master_bench.sh /tmp/setup_master_bench.sh --region $Region && chmod +x /tmp/setup_master_bench.sh && /tmp/setup_master_bench.sh")
} | ConvertTo-Json -Compress

$cmdResult = & $AWS ssm send-command `
    --instance-ids $masterInstanceId `
    --document-name "AWS-RunShellScript" `
    --parameters $ssmParams `
    --timeout-seconds 1800 `
    --region $Region `
    --output json 2>&1 | ConvertFrom-Json

$setupCmdId = $cmdResult.Command.CommandId
Write-Step "SETUP" "Setup command: $setupCmdId (building image, ~5-8 min)..."
$setupResult = Wait-SSMCommand $setupCmdId $masterInstanceId 15

if ($setupResult.Status -ne "Success") {
    Write-Step "SETUP" "Master setup failed!" "Red"
    Write-Host $setupResult.StandardErrorContent
    exit 1
}

# Get master private IP from output
$masterIp = ($setupResult.StandardOutputContent -split "`n" | Where-Object { $_ -match "MASTER_IP=" }) -replace "MASTER_IP=", ""
$masterIp = $masterIp.Trim()
Write-Step "SETUP" "Master IP: $masterIp" "Green"

# Setup GPU worker
Write-Step "SETUP" "Setting up worker instance..."
$workerSetupContent = @"
#!/bin/bash
set -e
export ARTIFACTS_BUCKET=$bucketName
export AWS_DEFAULT_REGION=$Region
MASTER_IP=$masterIp

mkdir -p /opt/spark-inference/app
aws s3 cp s3://$bucketName/inference/project.zip /opt/spark-inference/project.zip
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app
docker build --no-cache --network host -t multi-model-inference:latest -f deploy/Dockerfile .
docker rm -f spark-gpu-worker 2>/dev/null || true

# Try GPU mode first, fall back to CPU-only worker
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo '=== Starting GPU Worker ==='
  docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g -e NVIDIA_VISIBLE_DEVICES=all -v /opt/spark-inference/app/results:/app/results multi-model-inference:latest bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 8 -m 24g && tail -f /opt/spark/logs/*worker*"
else
  echo '=== Starting CPU Worker (no GPU detected) ==='
  docker run -d --name spark-gpu-worker --network host -v /opt/spark-inference/app/results:/app/results multi-model-inference:latest bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
fi

sleep 5
echo '=== Worker setup complete ==='
"@

$workerSetupPath = "$env:TEMP\setup_worker_bench.sh"
[System.IO.File]::WriteAllText($workerSetupPath, $workerSetupContent.Replace("`r`n", "`n"))
& $AWS s3 cp $workerSetupPath "s3://$bucketName/scripts/setup_worker_bench.sh" --region $Region

$workerSsmParams = @{
    commands = @("aws s3 cp s3://$bucketName/scripts/setup_worker_bench.sh /tmp/setup_worker_bench.sh --region $Region && chmod +x /tmp/setup_worker_bench.sh && /tmp/setup_worker_bench.sh")
} | ConvertTo-Json -Compress

$gpuCmdResult = & $AWS ssm send-command `
    --instance-ids $gpuWorkerInstanceId `
    --document-name "AWS-RunShellScript" `
    --parameters $workerSsmParams `
    --timeout-seconds 1800 `
    --region $Region `
    --output json 2>&1 | ConvertFrom-Json

$gpuSetupCmdId = $gpuCmdResult.Command.CommandId
Write-Step "SETUP" "Worker setup command: $gpuSetupCmdId..."
$gpuSetupResult = Wait-SSMCommand $gpuSetupCmdId $gpuWorkerInstanceId 15

if ($gpuSetupResult.Status -ne "Success") {
    Write-Step "SETUP" "Worker setup failed!" "Red"
}

# =============================================================================
# PHASE 3: RUN ALL BENCHMARK TESTS
# =============================================================================

Write-Host ""
Write-Step "BENCH" "--- Running complete benchmark suite ---"
Write-Step "BENCH" "This will take 60-90 minutes..."

$benchmarkScript = @"
#!/bin/bash
set -e
export SPARK_MASTER_URL=spark://$masterIp`:7077
export CUDA_VISIBLE_DEVICES=''
export RUN_TIMESTAMP=`$(date +%Y%m%d_%H%M%S)

echo '============================================================'
echo '  COMPLETE CPU TEST MATRIX — CLOUD RUN'
echo '============================================================'

cd /opt/spark-inference/app

# PHASE 1: MODE COMPARISON
echo '=== PHASE 1: MODE COMPARISON ==='
echo '  1.1 All modes - Small load'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=modes_small python benchmark/run_benchmark.py --mode all --signal-samples 1000 --image-samples 20 --detection-samples 5 --batch-size 64 --partitions 4" || echo 'PHASE 1.1 FAILED'

echo '  1.2 All modes - Medium load'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=modes_medium python benchmark/run_benchmark.py --mode all --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 4" || echo 'PHASE 1.2 FAILED'

echo '  1.3 Distributed - Large load'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=modes_large python benchmark/run_benchmark.py --mode distributed --signal-samples 10000 --image-samples 100 --detection-samples 20 --batch-size 64 --partitions 8" || echo 'PHASE 1.3 FAILED'

# PHASE 2: PARTITION SCALING
echo '=== PHASE 2: PARTITION SCALING ==='
for p in 2 4 6 8 12 16; do
    echo "  Partitions=`$p"
    docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=partitions_`$p python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions `$p" || echo "PHASE 2 p=`$p FAILED"
    sleep 3
done

# PHASE 3: DATA SIZE SCALING
echo '=== PHASE 3: DATA SIZE SCALING ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=datasize_tiny python benchmark/run_benchmark.py --mode distributed --signal-samples 500 --image-samples 10 --detection-samples 5 --batch-size 64 --partitions 4" || echo 'PHASE 3.1 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=datasize_small python benchmark/run_benchmark.py --mode distributed --signal-samples 1000 --image-samples 20 --detection-samples 5 --batch-size 64 --partitions 4" || echo 'PHASE 3.2 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=datasize_medium python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 4" || echo 'PHASE 3.3 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=datasize_large python benchmark/run_benchmark.py --mode distributed --signal-samples 8000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions 8" || echo 'PHASE 3.4 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=datasize_xlarge python benchmark/run_benchmark.py --mode distributed --signal-samples 10000 --image-samples 80 --detection-samples 15 --batch-size 64 --partitions 8" || echo 'PHASE 3.5 FAILED'
sleep 3

# PHASE 4: BATCH SIZE IMPACT
echo '=== PHASE 4: BATCH SIZE IMPACT ==='
for bs in 16 32 64 128 256 512; do
    echo "  Batch size=`$bs"
    docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=batch_`$bs python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size `$bs --partitions 4" || echo "PHASE 4 bs=`$bs FAILED"
    sleep 3
done

# PHASE 5: WORKER SCALING (dynamically add/remove CPU workers on master)
echo '=== PHASE 5: WORKER SCALING ==='
echo '  Scaling workers from 1 to 6 on the master node...'

# First stop the existing CPU worker
docker stop spark-cpu-worker 2>/dev/null || true
docker rm spark-cpu-worker 2>/dev/null || true
sleep 5

MASTER_IP=$masterIp
for w in 1 2 3 4 6; do
    echo "  5.$w Workers=$w — restarting workers..."
    
    # Stop all existing cpu worker containers
    for i in `$(seq 1 6); do
        docker stop spark-cpu-worker-`$i 2>/dev/null || true
        docker rm spark-cpu-worker-`$i 2>/dev/null || true
    done
    sleep 3
    
    # Start $w CPU workers
    for i in `$(seq 1 $w); do
        docker run -d --name spark-cpu-worker-`$i --network host \
          -v /opt/spark-inference/app/results:/app/results \
          multi-model-inference:latest \
          bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 2 -m 4g && tail -f /opt/spark/logs/*worker*"
    done
    
    # Wait for workers to register
    sleep 15
    
    PARTS=`$((w * 2))
    echo "    Running with $w workers, $PARTS partitions..."
    docker exec spark-master bash -c "SPARK_MASTER_URL=spark://`${MASTER_IP}:7077 CUDA_VISIBLE_DEVICES='' RUN_NAME=workers_`$w python benchmark/run_benchmark.py --mode distributed --signal-samples 5000 --image-samples 50 --detection-samples 10 --batch-size 64 --partitions `$PARTS" || echo "PHASE 5 w=`$w FAILED"
    sleep 5
done

# Restore: stop all scaling workers and restart the default CPU worker
for i in `$(seq 1 6); do
    docker stop spark-cpu-worker-`$i 2>/dev/null || true
    docker rm spark-cpu-worker-`$i 2>/dev/null || true
done
docker run -d --name spark-cpu-worker --network host \
  -v /opt/spark-inference/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
sleep 10
echo '  Worker scaling tests complete. Default cluster restored.'

# PHASE 6: CLUSTER BENCHMARK — CPU MODES
echo '=== PHASE 6: CLUSTER BENCHMARK (CPU) ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 2 --signal-samples 3000" || echo 'PHASE 6.1 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 3000" || echo 'PHASE 6.2 FAILED'
sleep 3
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 8 --signal-samples 3000" || echo 'PHASE 6.3 FAILED'
sleep 3

# PHASE 7: GPU BENCHMARK TESTS
echo '=== PHASE 7: GPU BENCHMARK TESTS ==='
echo '  7.1 gpu_only - small load (2 partitions, 1000 signals)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000 --image-samples 50 --detection-samples 20" || echo 'PHASE 7.1 FAILED'
sleep 3

echo '  7.2 gpu_only - medium load (4 partitions, 3000 signals)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30" || echo 'PHASE 7.2 FAILED'
sleep 3

echo '  7.3 gpu_only - large load (4 partitions, 5000 signals)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50" || echo 'PHASE 7.3 FAILED'
sleep 3

echo '  7.4 hybrid - small (GPU + CPU auto-detect, 2 partitions)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 2 --signal-samples 1000 --image-samples 50 --detection-samples 20" || echo 'PHASE 7.4 FAILED'
sleep 3

echo '  7.5 hybrid - medium (GPU + CPU auto-detect, 4 partitions)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30" || echo 'PHASE 7.5 FAILED'
sleep 3

echo '  7.6 hybrid - large (GPU + CPU auto-detect, 8 partitions)'
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 8 --signal-samples 5000 --image-samples 200 --detection-samples 50" || echo 'PHASE 7.6 FAILED'
sleep 3

# PHASE 8: GPU BATCH SIZE SCALING
echo '=== PHASE 8: GPU BATCH SIZE SCALING ==='
for bs in 32 64 128 256 512; do
    echo "  GPU batch_size=`$bs"
    docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size `$bs" || echo "PHASE 8 bs=`$bs FAILED"
    sleep 3
done

# PHASE 8b: GPU WORKER SCALING (add more CPU workers alongside the GPU worker)
echo '=== PHASE 8b: HYBRID WORKER SCALING (GPU + N CPU workers) ==='
MASTER_IP=$masterIp

for cpu_w in 0 1 2 4; do
    echo "  8b. GPU worker + $cpu_w CPU workers"
    
    # Stop all existing cpu worker containers
    docker stop spark-cpu-worker 2>/dev/null || true
    docker rm spark-cpu-worker 2>/dev/null || true
    for i in `$(seq 1 6); do
        docker stop spark-cpu-worker-`$i 2>/dev/null || true
        docker rm spark-cpu-worker-`$i 2>/dev/null || true
    done
    sleep 3
    
    # Start N CPU workers alongside the GPU worker
    for i in `$(seq 1 $cpu_w); do
        docker run -d --name spark-cpu-worker-`$i --network host \
          -v /opt/spark-inference/app/results:/app/results \
          multi-model-inference:latest \
          bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 2 -m 4g && tail -f /opt/spark/logs/*worker*"
    done
    sleep 10
    
    TOTAL_WORKERS=`$((cpu_w + 1))  # +1 for GPU worker
    PARTS=`$((TOTAL_WORKERS * 2))
    echo "    Total workers: `$TOTAL_WORKERS (1 GPU + `$cpu_w CPU), partitions: `$PARTS"
    docker exec spark-master bash -c "SPARK_MASTER_URL=spark://`${MASTER_IP}:7077 RUN_NAME=hybrid_scale_gpu1_cpu`${cpu_w} python benchmark/cluster_benchmark.py --device-mode hybrid --partitions `$PARTS --signal-samples 5000 --image-samples 100 --detection-samples 30" || echo "PHASE 8b cpu_w=`$cpu_w FAILED"
    sleep 5
done

# Restore default CPU worker
for i in `$(seq 1 6); do
    docker stop spark-cpu-worker-`$i 2>/dev/null || true
    docker rm spark-cpu-worker-`$i 2>/dev/null || true
done
docker run -d --name spark-cpu-worker --network host \
  -v /opt/spark-inference/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
sleep 10

# PHASE 9: INCREMENTAL LOAD TEST
echo '=== PHASE 9: INCREMENTAL LOAD TEST ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/incremental_load_test.py" || echo 'PHASE 9 FAILED'
sleep 5

# PHASE 10: FULL INCREMENTAL (all modes x 3 loads — includes gpu_only)
echo '=== PHASE 10: FULL INCREMENTAL (GPU + CPU + HYBRID x 3 loads) ==='
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://$masterIp`:7077 python benchmark/cluster_benchmark.py --incremental" || echo 'PHASE 10 FAILED'

echo '============================================================'
echo '  ALL TESTS COMPLETE'
echo '============================================================'

# Upload results to S3
echo '=== Uploading results to S3 ==='
docker cp spark-master:/app/results /opt/spark-inference/app/results/ 2>/dev/null || true
aws s3 sync /opt/spark-inference/app/results/ s3://$bucketName/results/ --region $Region
echo '=== Results uploaded ==='
"@

# Upload benchmark script to S3 and run via SSM
$benchScriptPath = "$env:TEMP\run_all_benchmarks.sh"
[System.IO.File]::WriteAllText($benchScriptPath, $benchmarkScript.Replace("`r`n", "`n"))
& $AWS s3 cp $benchScriptPath "s3://$bucketName/scripts/run_all_benchmarks.sh" --region $Region

$benchSsmParams = @{
    commands = @("aws s3 cp s3://$bucketName/scripts/run_all_benchmarks.sh /tmp/run_all_benchmarks.sh --region $Region && chmod +x /tmp/run_all_benchmarks.sh && /tmp/run_all_benchmarks.sh")
    executionTimeout = @("7200")
} | ConvertTo-Json -Compress

$benchCmdResult = & $AWS ssm send-command `
    --instance-ids $masterInstanceId `
    --document-name "AWS-RunShellScript" `
    --parameters $benchSsmParams `
    --timeout-seconds 7200 `
    --region $Region `
    --output json 2>&1 | ConvertFrom-Json

$benchCmdId = $benchCmdResult.Command.CommandId
Write-Step "BENCH" "Benchmark command: $benchCmdId"
Write-Step "BENCH" "Running all 8 phases... (est. 60-90 min)" "Yellow"

$benchResult = Wait-SSMCommand $benchCmdId $masterInstanceId 120

if ($benchResult.Status -eq "Success") {
    Write-Step "BENCH" "ALL BENCHMARKS COMPLETED SUCCESSFULLY!" "Green"
} else {
    Write-Step "BENCH" "Benchmark run had issues (partial results may exist)" "Yellow"
}

# =============================================================================
# PHASE 4: DOWNLOAD RESULTS
# =============================================================================

Write-Host ""
Write-Step "RESULTS" "--- Downloading results to local machine ---"

# Create local results directory
if (-not (Test-Path $ResultsLocalPath)) {
    New-Item -ItemType Directory -Path $ResultsLocalPath -Force | Out-Null
}

& $AWS s3 sync "s3://$bucketName/results/" $ResultsLocalPath --region $Region
Write-Step "RESULTS" "Results downloaded to: $ResultsLocalPath" "Green"

# =============================================================================
# PHASE 5: CLEANUP (terminate instances)
# =============================================================================

if (-not $KeepRunning) {
    Write-Host ""
    Write-Step "CLEANUP" "--- Terminating instances to stop billing ---"

    & $AWS cloudformation delete-stack --stack-name $STACK_NAME --region $Region
    Write-Step "CLEANUP" "Stack deletion initiated. Instances will terminate in ~2 min." "Green"
    Write-Step "CLEANUP" "To check: aws cloudformation describe-stacks --stack-name $STACK_NAME --region $Region"
} else {
    Write-Step "CLEANUP" "KeepRunning flag set — instances still running (you're being billed!)" "Yellow"
    $masterPublicIp = Get-StackOutput "MasterPublicIp"
    Write-Step "CLEANUP" "Spark UI: http://${masterPublicIp}:8080"
    Write-Step "CLEANUP" "To terminate: aws cloudformation delete-stack --stack-name $STACK_NAME --region $Region"
}

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "  BENCHMARK RUN COMPLETE" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""
Write-Step "DONE" "Results saved to: $ResultsLocalPath"
Write-Step "DONE" "Instance types used:"
Write-Step "DONE" "  Master: $MasterInstanceType (spot)"
Write-Step "DONE" "  GPU Worker: $WorkerInstanceType (spot)"
if (-not $KeepRunning) {
    Write-Step "DONE" "Instances terminated — no ongoing charges." "Green"
}
Write-Host ""
