# deploy_to_cluster.ps1 — One-command deploy to AWS Spark cluster
# Usage: .\deploy\deploy_to_cluster.ps1

$AWS = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$REGION = "ap-south-1"
$BUCKET = "sparkinferenceclusterstack-artifactsbucket2aac5544-zlcd7fgifndh"
$MASTER_ID = "i-075785f8565d7c606"
$GPU_WORKER_ID = "i-0e39cf399b1e9ef26"
$PROJECT_DIR = "D:\Spark_poc\simple_spark_torch_poc\pytorch-spark-inference-platform"

Set-Location $PROJECT_DIR

# --- Step 1: Zip (excluding junk) and upload ---
Write-Host "`n=== Step 1: Zip and upload ===" -ForegroundColor Cyan
if (Test-Path "project.zip") { Remove-Item "project.zip" -Force }
# Use Python to create zip with forward slashes (Linux-compatible)
python -c @"
import zipfile, os
exclude = {'.git', '.venv', 'cdk.out', 'node_modules', '__pycache__', 'project.zip'}
with zipfile.ZipFile('project.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if f == 'project.zip':
                continue
            filepath = os.path.join(root, f)
            arcname = filepath.replace('\\\\', '/')
            if arcname.startswith('./'):
                arcname = arcname[2:]
            zf.write(filepath, arcname)
    print(f'  Zipped {len(zf.namelist())} files')
"@
& $AWS s3 cp project.zip "s3://$BUCKET/inference/project.zip" --region $REGION
Write-Host "  Done" -ForegroundColor Green

# --- Step 2: Upload setup scripts ---
Write-Host "`n=== Step 2: Upload setup scripts ===" -ForegroundColor Cyan
& $AWS s3 cp deploy/scripts/setup_master.sh "s3://$BUCKET/scripts/setup_master.sh" --region $REGION
& $AWS s3 cp deploy/scripts/setup_gpu_worker.sh "s3://$BUCKET/scripts/setup_gpu_worker.sh" --region $REGION
Write-Host "  Done" -ForegroundColor Green

# --- Step 3: Run setup on Master ---
Write-Host "`n=== Step 3: Deploy to Master ===" -ForegroundColor Cyan
$masterJsonPath = "$env:TEMP\ssm_master.json"
[System.IO.File]::WriteAllText($masterJsonPath, '{"commands":["aws s3 cp s3://' + $BUCKET + '/scripts/setup_master.sh /tmp/setup_master.sh --region ' + $REGION + ' && chmod +x /tmp/setup_master.sh && /tmp/setup_master.sh"]}')
$masterResult = & $AWS ssm send-command --instance-ids $MASTER_ID --document-name "AWS-RunShellScript" --parameters "file://$masterJsonPath" --timeout-seconds 900 --region $REGION --output json 2>&1
if ($masterResult -match '"CommandId"') {
    $cmdId = ($masterResult | ConvertFrom-Json).Command.CommandId
    Write-Host "  Command sent: $cmdId" -ForegroundColor Green
    Write-Host "  Building on master (3-5 min)..." -ForegroundColor Yellow
    
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 30
        $status = & $AWS ssm get-command-invocation --command-id $cmdId --instance-id $MASTER_ID --region $REGION --output json 2>&1
        if ($status -match '"Success"') {
            Write-Host "  Master deploy SUCCESS" -ForegroundColor Green
            $parsed = $status | ConvertFrom-Json
            Write-Host $parsed.StandardOutputContent
            break
        } elseif ($status -match '"Failed"') {
            Write-Host "  Master deploy FAILED" -ForegroundColor Red
            $parsed = $status | ConvertFrom-Json
            Write-Host $parsed.StandardErrorContent
            break
        } elseif ($status -match '"InProgress"') {
            Write-Host "  Still building..." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  Failed: $masterResult" -ForegroundColor Red
}

# --- Step 4: Run setup on GPU Worker ---
Write-Host "`n=== Step 4: Deploy to GPU Worker ===" -ForegroundColor Cyan
$workerJsonPath = "$env:TEMP\ssm_worker.json"
[System.IO.File]::WriteAllText($workerJsonPath, '{"commands":["aws s3 cp s3://' + $BUCKET + '/scripts/setup_gpu_worker.sh /tmp/setup_gpu_worker.sh --region ' + $REGION + ' && chmod +x /tmp/setup_gpu_worker.sh && /tmp/setup_gpu_worker.sh"]}')
$workerResult = & $AWS ssm send-command --instance-ids $GPU_WORKER_ID --document-name "AWS-RunShellScript" --parameters "file://$workerJsonPath" --timeout-seconds 900 --region $REGION --output json 2>&1
if ($workerResult -match '"CommandId"') {
    $cmdId2 = ($workerResult | ConvertFrom-Json).Command.CommandId
    Write-Host "  Command sent: $cmdId2" -ForegroundColor Green
    Write-Host "  Building on GPU worker (3-5 min)..." -ForegroundColor Yellow
    
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 30
        $status = & $AWS ssm get-command-invocation --command-id $cmdId2 --instance-id $GPU_WORKER_ID --region $REGION --output json 2>&1
        if ($status -match '"Success"') {
            Write-Host "  GPU Worker deploy SUCCESS" -ForegroundColor Green
            $parsed = $status | ConvertFrom-Json
            Write-Host $parsed.StandardOutputContent
            break
        } elseif ($status -match '"Failed"') {
            Write-Host "  GPU Worker deploy FAILED" -ForegroundColor Red
            $parsed = $status | ConvertFrom-Json
            Write-Host $parsed.StandardErrorContent
            break
        } elseif ($status -match '"InProgress"') {
            Write-Host "  Still building..." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  Failed: $workerResult" -ForegroundColor Red
}

Write-Host "`n=== All Done ===" -ForegroundColor Green
Write-Host "Spark cluster should be running. Check Spark UI at the master's public IP:8080"
Write-Host "Run benchmark: SSM into master -> docker exec -it spark-master bash -c 'SPARK_MASTER_URL=spark://<IP>:7077 python benchmark/incremental_load_test.py'"
Write-Host ""
