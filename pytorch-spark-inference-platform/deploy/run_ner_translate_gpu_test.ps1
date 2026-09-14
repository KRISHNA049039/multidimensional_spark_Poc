# =============================================================================
# run_ner_translate_gpu_test.ps1 — Upload code, build+verify ner_translate on
# a real GPU instance, download the exported image + results back to local.
#
# Sibling to run_gpu_cdk.ps1 (same GpuBenchmarkStack, same S3/SSM pattern).
# Assumes GpuBenchmarkStack already deployed via CDK.
#
# Usage:
#   .\deploy\run_ner_translate_gpu_test.ps1
# =============================================================================
param(
    [string]$Region = "us-east-1",
    [string]$StackName = "GpuBenchmarkStack",
    [string]$ResultsLocalPath = ".\results\ner_translate_gpu",
    [string]$ImageLocalPath = ".\ner-translate-worker.tar.gz"
)

$ErrorActionPreference = "Stop"
$PROJECT_DIR = Split-Path -Parent $PSScriptRoot

function Write-Step($msg, $color = "Cyan") { Write-Host "  [NER-GPU] $msg" -ForegroundColor $color }

$bucket = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text
$instanceId = aws cloudformation describe-stacks --stack-name $StackName --region $Region --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text
Write-Step "Instance: $instanceId | Bucket: $bucket"

Write-Step "Zipping and uploading project (weights excluded — downloaded fresh on the instance, gitignored locally anyway)..."
Set-Location $PROJECT_DIR
if (Test-Path "project.zip") { Remove-Item "project.zip" -Force }

python -c @"
import zipfile, os
exclude = {'.git','.venv','cdk.out','node_modules','__pycache__','project.zip','results','wheels','wheels-hotfix'}
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
aws s3 cp deploy/scripts/build_and_test_ner_translate_gpu.sh "s3://$bucket/build_and_test_ner_translate_gpu.sh" --region $Region
Remove-Item "project.zip" -Force
Write-Step "Upload complete" "Green"

Write-Step "Starting build+test on EC2 (this includes downloading ~2-3GB of model weights — expect 15-25 min)..."
$ssmParams = @{
    commands = @("aws s3 cp s3://$bucket/build_and_test_ner_translate_gpu.sh /tmp/run.sh --region $Region && chmod +x /tmp/run.sh && /tmp/run.sh")
    executionTimeout = @("3600")
} | ConvertTo-Json -Compress

$cmd = aws ssm send-command --instance-ids $instanceId --document-name "AWS-RunShellScript" --parameters $ssmParams --timeout-seconds 3600 --region $Region --output json | ConvertFrom-Json
$cmdId = $cmd.Command.CommandId
Write-Step "SSM Command: $cmdId"
Write-Step "Live log: aws s3 cp s3://$bucket/ner_translate.log - --region $Region" "Yellow"

$elapsed = 0
$interval = 30
while ($elapsed -lt 3600) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval

    $status = aws ssm get-command-invocation --command-id $cmdId --instance-id $instanceId --region $Region --query "Status" --output text 2>&1

    if ($status.Trim() -eq "Success") {
        Write-Step "BUILD + VERIFY COMPLETE" "Green"
        break
    } elseif ($status.Trim() -eq "Failed") {
        Write-Step "Command reported FAILED — check the log before assuming the image is good" "Red"
        break
    } elseif ($status.Trim() -eq "TimedOut") {
        Write-Step "Command timed out" "Red"
        break
    }

    $min = [math]::Floor($elapsed / 60)
    $peekParams = '{"commands":["tail -1 /opt/benchmark/ner_translate.log 2>/dev/null"]}'
    $peekCmd = aws ssm send-command --instance-ids $instanceId --document-name "AWS-RunShellScript" --parameters $peekParams --region $Region --query "Command.CommandId" --output text 2>&1
    Start-Sleep -Seconds 8
    $peekOut = aws ssm get-command-invocation --command-id $peekCmd --instance-id $instanceId --region $Region --query "StandardOutputContent" --output text 2>&1
    Write-Step "${min}m | $peekOut" "Yellow"
}

Write-Step "Downloading exported image + results + log..."
if (-not (Test-Path $ResultsLocalPath)) { New-Item -ItemType Directory -Path $ResultsLocalPath -Force | Out-Null }
aws s3 cp "s3://$bucket/images/ner-translate-worker.tar.gz" $ImageLocalPath --region $Region
aws s3 sync "s3://$bucket/results/" $ResultsLocalPath --region $Region
aws s3 cp "s3://$bucket/ner_translate.log" "$ResultsLocalPath\ner_translate.log" --region $Region 2>$null

if (Test-Path $ImageLocalPath) {
    $sizeGB = [math]::Round((Get-Item $ImageLocalPath).Length / 1GB, 2)
    Write-Step "Downloaded verified image: $ImageLocalPath ($sizeGB GB)" "Green"
    Write-Step "Load locally with: docker load -i $ImageLocalPath"
} else {
    Write-Step "Image not found in S3 — the run likely failed before Step 6. Check $ResultsLocalPath\ner_translate.log" "Red"
}
Write-Step "Results: $ResultsLocalPath"
Write-Host ""
Write-Step "To destroy the instance and stop billing: npx cdk destroy $StackName --context region=$Region --context account=368287210840 --force"
