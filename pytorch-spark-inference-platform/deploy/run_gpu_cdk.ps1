# =============================================================================
# run_gpu_cdk.ps1 — Upload code, trigger benchmarks, stream logs, download results
# Assumes GpuBenchmarkStack already deployed via CDK
#
# Usage:
#   .\deploy\run_gpu_cdk.ps1
# =============================================================================
param(
    [string]$Region = "us-east-1",
    [string]$StackName = "GpuBenchmarkStack",
    [string]$ResultsLocalPath = ".\results\gpu_real"
)

$ErrorActionPreference = "Stop"
$PROJECT_DIR = Split-Path -Parent $PSScriptRoot

function Write-Step($msg, $color = "Cyan") { Write-Host "  [GPU] $msg" -ForegroundColor $color }

# Get stack outputs
$bucket = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text
$instanceId = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text
Write-Step "Instance: $instanceId | Bucket: $bucket"

# Upload project
Write-Step "Zipping and uploading project..."
Set-Location $PROJECT_DIR
if (Test-Path "project.zip") { Remove-Item "project.zip" -Force }

python -c @"
import zipfile, os
exclude = {'.git','.venv','cdk.out','node_modules','__pycache__','project.zip','results'}
with zipfile.ZipFile('project.zip','w',zipfile.ZIP_DEFLATED) as zf:
    for r,d,files in os.walk('.'):
        d[:] = [x for x in d if x not in exclude]
        for f in files:
            if f=='project.zip': continue
            p=os.path.join(r,f); a=p.replace('\\','/').lstrip('./')
            zf.write(p,a)
    print(f'  Zipped {len(zf.namelist())} files')
"@
aws s3 cp project.zip "s3://$bucket/project.zip" --region $Region
aws s3 cp deploy/scripts/setup_and_run_gpu.sh "s3://$bucket/setup_and_run_gpu.sh" --region $Region
Remove-Item "project.zip" -Force
Write-Step "Upload complete" "Green"

# Trigger the benchmark script via SSM
Write-Step "Starting benchmark on EC2..."
$ssmParams = @{
    commands = @("aws s3 cp s3://$bucket/setup_and_run_gpu.sh /tmp/run.sh --region $Region && chmod +x /tmp/run.sh && /tmp/run.sh")
    executionTimeout = @("7200")
} | ConvertTo-Json -Compress

$cmd = aws ssm send-command --instance-ids $instanceId --document-name "AWS-RunShellScript" --parameters $ssmParams --timeout-seconds 7200 --region $Region --output json | ConvertFrom-Json
$cmdId = $cmd.Command.CommandId
Write-Step "SSM Command: $cmdId"
Write-Step "Monitoring progress (check logs with: aws s3 cp s3://$bucket/benchmark.log - --region $Region)" "Yellow"

# Poll for completion with live status
$elapsed = 0
$interval = 30
while ($elapsed -lt 7200) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval

    $status = aws ssm get-command-invocation --command-id $cmdId --instance-id $instanceId --region $Region --query "Status" --output text 2>&1

    if ($status.Trim() -eq "Success") {
        Write-Step "BENCHMARKS COMPLETED!" "Green"
        break
    } elseif ($status.Trim() -eq "Failed") {
        Write-Step "Command reported FAILED (may have partial results)" "Yellow"
        break
    } elseif ($status.Trim() -eq "TimedOut") {
        Write-Step "Command timed out" "Red"
        break
    }

    # Live status peek
    $min = [math]::Floor($elapsed / 60)
    $peekParams = '{"commands":["ls /opt/benchmark/app/results/*.json 2>/dev/null | wc -l; ls -t /opt/benchmark/app/results/*.json 2>/dev/null | head -1; tail -1 /opt/benchmark/benchmark.log 2>/dev/null"]}'
    $peekCmd = aws ssm send-command --instance-ids $instanceId --document-name "AWS-RunShellScript" --parameters $peekParams --region $Region --query "Command.CommandId" --output text 2>&1
    Start-Sleep -Seconds 8
    $peekOut = aws ssm get-command-invocation --command-id $peekCmd --instance-id $instanceId --region $Region --query "StandardOutputContent" --output text 2>&1
    $peekStr = [string]$peekOut
    $lines = $peekStr -split "`n" | Where-Object { [string]$_ -ne "" }
    $fileCount = if ($lines.Count -gt 0) { [string]$lines[0] } else { "?" }
    $latestFile = if ($lines.Count -gt 1) { [string]$lines[1] | Split-Path -Leaf } else { "" }
    $logTail = if ($lines.Count -gt 2) { [string]$lines[2] } else { "" }
    Write-Step "${min}m | Files: $fileCount | Latest: $latestFile | $logTail" "Yellow"
}

# Download results
Write-Step "Downloading results..."
if (-not (Test-Path $ResultsLocalPath)) { New-Item -ItemType Directory -Path $ResultsLocalPath -Force | Out-Null }
aws s3 sync "s3://$bucket/results/" $ResultsLocalPath --region $Region
# Also download the full log
aws s3 cp "s3://$bucket/benchmark.log" "$ResultsLocalPath\benchmark.log" --region $Region 2>$null

$count = (Get-ChildItem $ResultsLocalPath -Filter "*.json" -ErrorAction SilentlyContinue).Count
Write-Step "Downloaded $count result files to: $ResultsLocalPath" "Green"
Write-Step "Full log: $ResultsLocalPath\benchmark.log"
Write-Host ""
Write-Step "To destroy: npx cdk destroy $StackName --context region=$Region --context account=368287210840 --force"
