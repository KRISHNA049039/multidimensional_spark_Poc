# =============================================================================
# run_gpu_benchmarks.ps1 — Run Phases 6-10 (GPU + hybrid) with fixed config
#
# Fixes from Phase 6 deadlock:
#   - executor memory: 4g (was 2g — OOM with 10 models)
#   - executor cores: 4 with task.cpus=2 (fewer concurrent model loads)
#   - python worker memory: 4g
#   - dynamic partition count based on available cores
#
# Prerequisites:
#   - GPU quota approved (G/VT instance limit >= 8 vCPUs)
#   - AWS CLI configured
#
# Usage:
#   .\deploy\run_gpu_benchmarks.ps1
#   .\deploy\run_gpu_benchmarks.ps1 -SkipDeploy    # Use existing stack
# =============================================================================

param(
    [string]$Region = "us-east-1",
    [string]$MasterInstanceType = "c5.2xlarge",
    [string]$WorkerInstanceType = "g4dn.2xlarge",
    [string]$ResultsLocalPath = ".\results\gpu_benchmark",
    [switch]$SkipDeploy,
    [switch]$KeepRunning
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
    $interval = 20
    while ($elapsed -lt ($timeoutMin * 60)) {
        Start-Sleep -Seconds $interval
        $elapsed += $interval

        $status = & $AWS ssm get-command-invocation `
            --command-id $commandId `
            --instance-id $instanceId `
            --region $Region `
            --query "Status" --output text 2>&1

        switch ($status.Trim()) {
            "Success" {
                Write-Step "SSM" "Command completed" "Green"
                $output = & $AWS ssm get-command-invocation `
                    --command-id $commandId --instance-id $instanceId `
                    --region $Region --query "StandardOutputContent" --output text 2>&1
                return @{ Status = "Success"; StandardOutputContent = $output }
            }
            "Failed" {
                Write-Step "SSM" "Command FAILED" "Red"
                $errOutput = & $AWS ssm get-command-invocation `
                    --command-id $commandId --instance-id $instanceId `
                    --region $Region --query "StandardErrorContent" --output text 2>&1
                Write-Host $errOutput
                return @{ Status = "Failed"; StandardErrorContent = $errOutput }
            }
            "TimedOut" {
                Write-Step "SSM" "Command timed out" "Red"
                return @{ Status = "TimedOut" }
            }
            default {
                $min = [math]::Floor($elapsed / 60)
                # Poll live status from the instance
                $liveParams = '{"commands":["docker exec spark-master ls /app/results/ 2>/dev/null | wc -l; docker exec spark-master ls -t /app/results/ 2>/dev/null | head -3; docker ps --format \"table {{.Names}}\\t{{.Status}}\" 2>/dev/null | head -5"]}'
                $peekCmd = & $AWS ssm send-command --instance-ids $instanceId `
                    --document-name "AWS-RunShellScript" --parameters $liveParams `
                    --region $Region --query "Command.CommandId" --output text 2>&1
                Start-Sleep -Seconds 8
                $peekOut = & $AWS ssm get-command-invocation --command-id $peekCmd `
                    --instance-id $instanceId --region $Region `
                    --query "StandardOutputContent" --output text 2>&1

                # Parse and display live info
                $lines = ($peekOut -split "`n") | Where-Object { $_.Trim() -ne "" }
                $fileCount = if ($lines.Count -gt 0) { $lines[0].Trim() } else { "?" }
                $latestFile = if ($lines.Count -gt 1) { $lines[1].Trim() } else { "" }

                Write-Step "LIVE" "${min}m elapsed | Results: $fileCount files | Latest: $latestFile" "Yellow"
            }
        }
    }
    return $null
}

function Get-StackOutput($outputKey) {
    & $AWS cloudformation describe-stacks `
        --stack-name $STACK_NAME --region $Region `
        --query "Stacks[0].Outputs[?OutputKey=='$outputKey'].OutputValue" `
        --output text 2>&1
}

# =============================================================================
# DEPLOY OR REUSE STACK
# =============================================================================

Write-Host ""
Write-Host "  GPU BENCHMARK — Phases 6-10" -ForegroundColor White
Write-Host "  Master: $MasterInstanceType | Worker: $WorkerInstanceType" -ForegroundColor Gray
Write-Host ""

if (-not $SkipDeploy) {
    Write-Step "DEPLOY" "Deploying stack with GPU worker..."

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

    $templatePath = Join-Path $PROJECT_DIR "deploy\spot_benchmark_cfn.yaml"
    & $AWS cloudformation deploy `
        --template-file $templatePath `
        --stack-name $STACK_NAME --region $Region `
        --capabilities CAPABILITY_IAM `
        --parameter-overrides MasterInstanceType=$MasterInstanceType WorkerInstanceType=$WorkerInstanceType `
        --no-fail-on-empty-changeset

    $bucketName = Get-StackOutput "ArtifactsBucketName"
    & $AWS s3 cp project.zip "s3://$bucketName/inference/project.zip" --region $Region
    Remove-Item "project.zip" -Force

    $masterInstanceId = Get-StackOutput "MasterInstanceId"
    $gpuWorkerInstanceId = Get-StackOutput "GpuWorkerInstanceId"

    # Wait for SSM
    Write-Step "DEPLOY" "Waiting for SSM..."
    $retries = 0
    do {
        Start-Sleep -Seconds 15
        $retries++
        $ssmStatus = & $AWS ssm describe-instance-information `
            --filters "Key=InstanceIds,Values=$masterInstanceId" `
            --region $Region --output json 2>&1 | ConvertFrom-Json
        $online = $ssmStatus.InstanceInformationList.Count -gt 0
    } while (-not $online -and $retries -lt 40)

    Write-Step "DEPLOY" "Stack ready" "Green"
} else {
    $bucketName = Get-StackOutput "ArtifactsBucketName"
    $masterInstanceId = Get-StackOutput "MasterInstanceId"
    $gpuWorkerInstanceId = Get-StackOutput "GpuWorkerInstanceId"

    # Re-upload latest code
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
    Write-Step "DEPLOY" "Using existing stack, code re-uploaded"
}

# =============================================================================
# SETUP: Build image + start cluster with PROPER memory settings
# =============================================================================

Write-Step "SETUP" "Deploying with fixed Spark config (4GB executor memory)..."

$masterSetup = @"
#!/bin/bash
set -e
aws s3 cp s3://$bucketName/inference/project.zip /opt/spark-inference/project.zip --region $Region
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app

docker build --no-cache --network host -t multi-model-inference:latest -f deploy/Dockerfile .

docker rm -f spark-master spark-cpu-worker 2>/dev/null || true
MASTER_IP=`$(hostname -I | awk '{print `$1}')

# Master with higher driver memory
docker run -d --name spark-master --network host \
  -e SPARK_DRIVER_MEMORY=8g \
  -v /opt/spark-inference/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"

sleep 10

# CPU worker on master — 4 cores, 12GB (allows 2 executors × 4GB + overhead)
docker run -d --name spark-cpu-worker --network host \
  -v /opt/spark-inference/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://`${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"

sleep 5
echo "MASTER_IP=`${MASTER_IP}"
echo '=== Master ready ==='
"@

$masterSetupPath = "$env:TEMP\gpu_setup_master.sh"
[System.IO.File]::WriteAllText($masterSetupPath, $masterSetup.Replace("`r`n", "`n"))
& $AWS s3 cp $masterSetupPath "s3://$bucketName/scripts/gpu_setup_master.sh" --region $Region

$ssmParams = @{ commands = @("aws s3 cp s3://$bucketName/scripts/gpu_setup_master.sh /tmp/s.sh --region $Region && chmod +x /tmp/s.sh && /tmp/s.sh") } | ConvertTo-Json -Compress
$cmd = & $AWS ssm send-command --instance-ids $masterInstanceId --document-name "AWS-RunShellScript" --parameters $ssmParams --timeout-seconds 1800 --region $Region --output json 2>&1 | ConvertFrom-Json
$result = Wait-SSMCommand $cmd.Command.CommandId $masterInstanceId 20

$masterIp = ($result.StandardOutputContent -split "`n" | Where-Object { $_ -match "MASTER_IP=" }) -replace "MASTER_IP=", ""
$masterIp = $masterIp.Trim()
Write-Step "SETUP" "Master IP: $masterIp" "Green"

# GPU Worker setup
$workerSetup = @"
#!/bin/bash
set -e
aws s3 cp s3://$bucketName/inference/project.zip /opt/spark-inference/project.zip --region $Region
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app
cd /opt/spark-inference/app
docker build --no-cache --network host -t multi-model-inference:latest -f deploy/Dockerfile .
docker rm -f spark-gpu-worker 2>/dev/null || true

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /opt/spark-inference/app/results:/app/results \
    multi-model-inference:latest \
    bash -c "start-worker.sh spark://$masterIp:7077 -c 8 -m 24g && tail -f /opt/spark/logs/*worker*"
else
  docker run -d --name spark-gpu-worker --network host \
    -v /opt/spark-inference/app/results:/app/results \
    multi-model-inference:latest \
    bash -c "start-worker.sh spark://$masterIp:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
fi
sleep 5
echo '=== Worker ready ==='
"@

$workerSetupPath = "$env:TEMP\gpu_setup_worker.sh"
[System.IO.File]::WriteAllText($workerSetupPath, $workerSetup.Replace("`r`n", "`n"))
& $AWS s3 cp $workerSetupPath "s3://$bucketName/scripts/gpu_setup_worker.sh" --region $Region

$wParams = @{ commands = @("aws s3 cp s3://$bucketName/scripts/gpu_setup_worker.sh /tmp/w.sh --region $Region && chmod +x /tmp/w.sh && /tmp/w.sh") } | ConvertTo-Json -Compress
$wcmd = & $AWS ssm send-command --instance-ids $gpuWorkerInstanceId --document-name "AWS-RunShellScript" --parameters $wParams --timeout-seconds 1800 --region $Region --output json 2>&1 | ConvertFrom-Json
Wait-SSMCommand $wcmd.Command.CommandId $gpuWorkerInstanceId 20

# =============================================================================
# RUN GPU BENCHMARKS (Phases 6-10) with fixed Spark config
# =============================================================================

Write-Step "BENCH" "Running GPU benchmark phases..."

# Use static benchmark script — replace placeholders with actual values
$benchScriptTemplate = Join-Path $PROJECT_DIR "deploy\scripts\gpu_benchmarks.sh"
$benchScriptContent = [System.IO.File]::ReadAllText($benchScriptTemplate)
$benchScriptContent = $benchScriptContent.Replace("__MASTER_IP__", $masterIp)
$benchScriptContent = $benchScriptContent.Replace("__BUCKET__", $bucketName)
$benchScriptContent = $benchScriptContent.Replace("__REGION__", $Region)

$benchPath = "$env:TEMP\gpu_benchmarks.sh"
[System.IO.File]::WriteAllText($benchPath, $benchScriptContent.Replace("`r`n", "`n"))
& $AWS s3 cp $benchPath "s3://$bucketName/scripts/gpu_benchmarks.sh" --region $Region

$bParams = @{ commands = @("aws s3 cp s3://$bucketName/scripts/gpu_benchmarks.sh /tmp/b.sh --region $Region && chmod +x /tmp/b.sh && /tmp/b.sh"); executionTimeout = @("7200") } | ConvertTo-Json -Compress
$bcmd = & $AWS ssm send-command --instance-ids $masterInstanceId --document-name "AWS-RunShellScript" --parameters $bParams --timeout-seconds 7200 --region $Region --output json 2>&1 | ConvertFrom-Json

Write-Step "BENCH" "Command: $($bcmd.Command.CommandId)"
Write-Step "BENCH" "Running phases 6-10... (~45-60 min)" "Yellow"

$benchResult = Wait-SSMCommand $bcmd.Command.CommandId $masterInstanceId 90

# =============================================================================
# DOWNLOAD RESULTS
# =============================================================================

if (-not (Test-Path $ResultsLocalPath)) {
    New-Item -ItemType Directory -Path $ResultsLocalPath -Force | Out-Null
}

# Always try to sync results even if benchmark "failed" (Spark logs cause false failures)
Write-Step "RESULTS" "Downloading results from S3..."
& $AWS s3 sync "s3://$bucketName/results/" $ResultsLocalPath --region $Region

$resultCount = (Get-ChildItem -Path $ResultsLocalPath -Filter "*.json" -ErrorAction SilentlyContinue).Count
Write-Step "RESULTS" "Downloaded $resultCount JSON files to: $ResultsLocalPath" "Green"

# =============================================================================
# CLEANUP — only delete if we got results, otherwise keep for debugging
# =============================================================================

if ($resultCount -gt 0 -and -not $KeepRunning) {
    & $AWS cloudformation delete-stack --stack-name $STACK_NAME --region $Region
    Write-Step "CLEANUP" "Stack deletion initiated" "Green"
} elseif ($resultCount -eq 0) {
    Write-Step "CLEANUP" "No results downloaded! Stack kept alive for debugging." "Red"
    Write-Step "CLEANUP" "Check manually: aws ssm start-session --target $masterInstanceId --region $Region" "Yellow"
    Write-Step "CLEANUP" "To delete later: aws cloudformation delete-stack --stack-name $STACK_NAME --region $Region" "Yellow"
} else {
    Write-Step "CLEANUP" "Stack kept running (billing continues!)" "Yellow"
}

Write-Host ""
Write-Step "DONE" "GPU benchmark complete. Results in: $ResultsLocalPath ($resultCount files)"
Write-Host ""
